"""
models/aod_beer_lambert.py

Pollution-aware attenuation model: Beer-Lambert extinction driven by
measured Aerosol Optical Depth (designed around MODIS MAIAC 550nm, but
format-agnostic -- it consumes whatever io.aod_loaders hands it).

    transmission(t) = exp(-AOD(t) * m(t))

where m(t) is the relative (geometric) optical air mass at the solar
zenith angle -- not the pressure-corrected absolute air mass. That choice
is deliberate: AOD is itself a directly-retrieved column quantity, not a
surface-pressure-scaled Rayleigh term, so the standard practice for
slant-path aerosol corrections (e.g. AERONET-style processing) is to use
the plain geometric air mass; pressure correction belongs to Rayleigh
optical depth, which does scale with the amount of air present.

Scientific caveat (worth restating in a Methodology section)
--------------------------------------------------------------
The clear-sky DNI this transmission multiplies (solar.py) already comes
from the Ineichen model driven by a *climatological* Linke turbidity,
which itself represents a typical long-term aerosol + water-vapor load.
Multiplying by a second, MAIAC-derived aerosol transmission term is
therefore a first-order approximation of the *excess* attenuation from a
specific pollution episode relative to that climatology, not a from-
scratch radiative-transfer solve -- there is a theoretical risk of
partially double-counting "typical" background aerosol content baked into
the turbidity climatology. For a stricter decomposition, set
``solar.linke_turbidity_source: "fixed"`` with a low, near-Rayleigh
``fixed_linke_turbidity`` (e.g. 1.5-2.0) so the clear-sky baseline carries
minimal built-in aerosol assumption, leaving (nearly) all aerosol
attenuation to this model.

Spatial grouping
------------------
Rather than storing a full (n_faces, n_hours) transmission array -- which
would be almost entirely duplicate rows for a compact AOI spanning a
handful of ~1 km AOD pixels -- every simulated face is assigned to its
*nearest* AOD pixel by great-circle-ish (flat-Earth-approximated)
distance, and only the *touched* pixels become "groups". Transmission is
computed once per group; simulation.py expands group -> face via
``face_to_group_id`` only at the point irradiance is actually multiplied
out, never in storage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from ..config import AODIngestConfig, LocationConfig
from ..constants import MODEL_KEY_AOD_BEER_LAMBERT
from ..geometry import CityGeometry
from ..io.aod_loaders import AODDataset
from ..utilities import enu_to_latlon, get_logger
from .base import BaseAttenuationModel, register_model

logger = get_logger()


def _interpolate_one_series(
    raw: pd.Series,
    target_index: pd.DatetimeIndex,
    max_gap_hours: float,
    fallback_value: float,
) -> tuple[np.ndarray, int]:
    """Interpolate one pixel's irregular-cadence observations onto
    `target_index` (the full simulation hourly index), returning
    (values, n_fallback_hours).

    Two-step approach: (1) time-weighted linear interpolation across the
    union of observation + target timestamps, with flat extrapolation
    at the boundaries via ``limit_direction="both"`` so nothing is left
    NaN; (2) any target hour whose nearest real observation is farther
    than `max_gap_hours` away (this catches both large interior gaps and
    boundary extrapolation stretched too far) is overridden with
    `fallback_value` instead of trusting a stale interpolated/extrapolated
    guess, and counted so the caller can log a single summary warning.
    """
    raw = raw.dropna()
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()

    if len(raw) == 0:
        return np.full(len(target_index), fallback_value, dtype=np.float64), len(target_index)

    union_index = raw.index.union(target_index)
    combined = pd.Series(index=union_index, dtype=np.float64)
    combined.loc[raw.index] = raw.to_numpy()
    interpolated = combined.interpolate(method="time", limit_direction="both")
    result = interpolated.reindex(target_index).to_numpy()

    valid_t = raw.index.values  # UTC-normalized datetime64, tz-agnostic for distance purposes
    target_t = target_index.values
    order = np.argsort(valid_t)
    vt_sorted = valid_t[order]
    pos = np.searchsorted(vt_sorted, target_t)
    pos_right = np.clip(pos, 0, len(vt_sorted) - 1)
    pos_left = np.clip(pos - 1, 0, len(vt_sorted) - 1)
    gap_right = np.abs(vt_sorted[pos_right] - target_t)
    gap_left = np.abs(vt_sorted[pos_left] - target_t)
    nearest_gap_hours = np.minimum(gap_left, gap_right) / np.timedelta64(1, "h")

    needs_fallback = nearest_gap_hours > max_gap_hours
    result = np.where(needs_fallback | np.isnan(result), fallback_value, result)
    return result.astype(np.float64), int(needs_fallback.sum())


@register_model(MODEL_KEY_AOD_BEER_LAMBERT)
class AODAttenuationModel(BaseAttenuationModel):
    """Beer-Lambert AOD attenuation model. See module docstring for the
    physics and the spatial-grouping design.
    """

    name = MODEL_KEY_AOD_BEER_LAMBERT

    def __init__(
        self,
        aod_dataset: AODDataset,
        geometry: CityGeometry,
        location_cfg: LocationConfig,
        ingest_cfg: AODIngestConfig,
        full_time_index: pd.DatetimeIndex,
        airmass_model: str,
    ):
        self._airmass_model = airmass_model
        self._full_times = full_time_index

        self._face_to_group_id, touched_pixels, self.group_lat, self.group_lon = self._assign_groups(
            aod_dataset, geometry, location_cfg
        )
        self.n_groups = len(touched_pixels)

        group_values = np.empty((self.n_groups, len(full_time_index)), dtype=np.float64)
        n_fallback_total = 0
        for g, pixel_idx in enumerate(touched_pixels):
            raw_series = pd.Series(aod_dataset.values[pixel_idx, :], index=aod_dataset.times)
            values, n_fallback = _interpolate_one_series(
                raw_series, full_time_index, ingest_cfg.max_gap_hours, ingest_cfg.climatological_fallback_value
            )
            group_values[g] = values
            n_fallback_total += n_fallback

        if n_fallback_total:
            logger.warning(
                f"{n_fallback_total} group-hour(s) across {self.n_groups} AOD group(s) fell further than "
                f"{ingest_cfg.max_gap_hours}h from the nearest real observation; used the climatological "
                f"fallback value ({ingest_cfg.climatological_fallback_value})"
            )

        self._group_aod_full = pd.DataFrame(group_values.T, index=full_time_index, columns=range(self.n_groups))
        logger.info(
            f"AODAttenuationModel ready: {self.n_groups} spatial group(s), "
            f"AOD range used [{group_values.min():.3f}, {group_values.max():.3f}]"
        )

    # ------------------------------------------------------------------
    def _assign_groups(
        self, aod_dataset: AODDataset, geometry: CityGeometry, location_cfg: LocationConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (face_to_group_id, touched_pixel_indices, group_lat, group_lon)."""
        if not aod_dataset.spatially_varying or aod_dataset.n_pixels == 1:
            face_to_group = np.zeros(geometry.n_faces, dtype=np.int32)
            touched_pixels = np.array([0], dtype=np.int64)
            return face_to_group, touched_pixels, np.array([location_cfg.latitude]), np.array([location_cfg.longitude])

        face_lat, face_lon = enu_to_latlon(
            geometry.centroid[:, 0], geometry.centroid[:, 1], location_cfg.latitude, location_cfg.longitude
        )
        # Small n_pixels expected for a district-scale AOI (a handful of ~1km cells) --
        # brute-force nearest-neighbor keeps this dependency-free and is negligible cost.
        d2 = (aod_dataset.lat[None, :] - face_lat[:, None]) ** 2 + (aod_dataset.lon[None, :] - face_lon[:, None]) ** 2
        nearest_pixel = np.argmin(d2, axis=1)
        touched_pixels = np.unique(nearest_pixel)
        remap = {orig: new for new, orig in enumerate(touched_pixels)}
        face_to_group = np.array([remap[p] for p in nearest_pixel], dtype=np.int32)

        logger.info(
            f"Spatially-varying AOD: {len(touched_pixels)} of {aod_dataset.n_pixels} pixel(s) "
            f"touched by the city footprint"
        )
        return face_to_group, touched_pixels, aod_dataset.lat[touched_pixels], aod_dataset.lon[touched_pixels]

    # ------------------------------------------------------------------
    def compute_transmission(self, times: pd.DatetimeIndex, solar_zenith_deg: np.ndarray) -> np.ndarray:
        """Beer-Lambert transmission for each group, for the given hour
        chunk. `times` must be a contiguous subset of the full simulation
        time index this model was constructed with.
        """
        aod_chunk = self._group_aod_full.reindex(times).to_numpy().T  # (n_groups, n_hours)
        airmass_rel = pvlib.atmosphere.get_relative_airmass(
            pd.Series(solar_zenith_deg, index=times), model=self._airmass_model
        ).to_numpy()
        # Air mass is only physically meaningful (and only ever multiplied against a
        # nonzero DNI) above the horizon; night-hour NaN/huge values from the airmass
        # formula must never propagate into a NaN transmission.
        airmass_rel = np.nan_to_num(airmass_rel, nan=0.0, posinf=0.0)
        transmission = np.exp(-aod_chunk * airmass_rel[None, :])
        return np.clip(transmission, 0.0, 1.0).astype(np.float32)

    def group_metadata(self) -> dict:
        return {
            "aod_group_lat": self.group_lat.astype(np.float32),
            "aod_group_lon": self.group_lon.astype(np.float32),
            "aod_group_value": self._group_aod_full.to_numpy().T.astype(np.float32),  # (n_groups, n_hours)
        }

    def face_to_group_id(self, n_faces: int) -> np.ndarray:
        if n_faces != len(self._face_to_group_id):
            raise ValueError(
                f"AODAttenuationModel was built for {len(self._face_to_group_id)} faces, "
                f"but face_to_group_id({n_faces}) was requested -- geometry mismatch"
            )
        return self._face_to_group_id
