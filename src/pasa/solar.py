"""
solar.py

Sun position and clear-sky direct normal irradiance (DNI) for the whole
simulated period, computed once at the AOI's single reference location
(config.location) rather than per-face. This is a deliberate, standard
simplifying assumption: for a district-scale AOI (a few km across, such as
Civil Lines), the difference in apparent solar position between the near
and far edges of the site is well under the angular resolution needed for
hourly irradiance modelling. If PASA-Sim is ever pointed at a much larger
domain, this assumption should be revisited (e.g. compute solar position
per building cluster).

Everything here is either a direct pvlib call or simple, well-established
trigonometry -- there are no fitted/hardcoded "typical" values; Linke
turbidity comes from pvlib's bundled climatology (or an explicit,
user-chosen fixed override -- see config.SolarConfig), and extraterrestrial
radiation comes from the Spencer (1971) Fourier-series formula via pvlib,
not a flat solar-constant stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

from .config import LocationConfig, SolarConfig, TimeConfig
from .utilities import get_logger

logger = get_logger()


@dataclass
class SolarTimeSeries:
    """Per-hour solar geometry and clear-sky irradiance, aligned on `times`."""

    times: pd.DatetimeIndex
    zenith_deg: np.ndarray  # apparent zenith, (n_hours,) float32
    azimuth_deg: np.ndarray  # (n_hours,) float32, pvlib convention (0=N,90=E,180=S,270=W)
    elevation_deg: np.ndarray  # apparent elevation, (n_hours,) float32
    dni_clear_w_m2: np.ndarray  # (n_hours,) float32, Ineichen clear-sky DNI
    dni_extra_w_m2: np.ndarray  # (n_hours,) float32, extraterrestrial DNI (Spencer)
    sun_vector_enu: np.ndarray  # (n_hours, 3) float32, unit vector FROM surface TOWARD sun

    @property
    def n_hours(self) -> int:
        return len(self.times)


def build_time_index(time_cfg: TimeConfig, tz: str) -> pd.DatetimeIndex:
    """Construct the simulation's hourly time index. Leap years are handled
    automatically by pandas' calendar arithmetic (Dec 31 of a leap year is
    still Dec 31, `pd.date_range` naturally yields 8784 hourly timestamps
    instead of 8760) -- there is no leap-year special case anywhere in this
    codebase, and no file should ever hardcode 8760 as an array length.
    """
    if time_cfg.start and time_cfg.end:
        start, end = time_cfg.start, time_cfg.end
    else:
        start = f"{time_cfg.year}-01-01 00:00"
        end = f"{time_cfg.year}-12-31 23:00"
    times = pd.date_range(start=start, end=end, freq=time_cfg.freq, tz=tz)
    logger.info(f"Time index: {times[0]} to {times[-1]} ({len(times):,} hours)")
    return times


def compute_solar_time_series(
    times: pd.DatetimeIndex,
    location_cfg: LocationConfig,
    solar_cfg: SolarConfig,
) -> SolarTimeSeries:
    """Compute sun position, clear-sky DNI, and ENU sun-direction vectors
    for every timestamp in `times`.
    """
    location = pvlib.location.Location(
        latitude=location_cfg.latitude,
        longitude=location_cfg.longitude,
        tz=location_cfg.timezone,
        altitude=location_cfg.altitude_m,
    )
    solpos = location.get_solarposition(times)

    if solar_cfg.linke_turbidity_source == "climatology":
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
            times, location_cfg.latitude, location_cfg.longitude
        )
    else:
        linke_turbidity = pd.Series(solar_cfg.fixed_linke_turbidity, index=times)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass_rel = pvlib.atmosphere.get_relative_airmass(
        solpos["apparent_zenith"], model=solar_cfg.airmass_model
    )
    pressure = pvlib.atmosphere.alt2pres(location_cfg.altitude_m)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)

    clearsky = pvlib.clearsky.ineichen(
        apparent_zenith=solpos["apparent_zenith"],
        airmass_absolute=airmass_abs,
        linke_turbidity=linke_turbidity,
        altitude=location_cfg.altitude_m,
        dni_extra=dni_extra,
    )

    elevation_deg = solpos["apparent_elevation"].to_numpy(dtype=np.float32)
    zenith_deg = solpos["apparent_zenith"].to_numpy(dtype=np.float32)
    azimuth_deg = solpos["azimuth"].to_numpy(dtype=np.float32)
    dni_clear = clearsky["dni"].to_numpy(dtype=np.float32)

    # Defensive clip: force DNI to exactly 0 below the horizon regardless of
    # any twilight-adjacent numerical artifact from the underlying airmass
    # formula (Ineichen already does this internally in practice, but the
    # explicit clip makes the invariant true by construction rather than by
    # incidental library behavior).
    below_horizon = elevation_deg <= 0.0
    dni_clear = np.where(below_horizon, 0.0, dni_clear).astype(np.float32)
    dni_clear = np.clip(dni_clear, 0.0, None)

    sun_vector_enu = _solar_angles_to_enu_vector(elevation_deg, azimuth_deg)

    logger.info(
        f"Solar geometry computed for {len(times):,} hours "
        f"(peak clear-sky DNI: {dni_clear.max():.0f} W/m^2, "
        f"{int(below_horizon.sum()):,} night hours)"
    )

    return SolarTimeSeries(
        times=times,
        zenith_deg=zenith_deg,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        dni_clear_w_m2=dni_clear,
        dni_extra_w_m2=dni_extra.to_numpy(dtype=np.float32),
        sun_vector_enu=sun_vector_enu,
    )


def _solar_angles_to_enu_vector(elevation_deg: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
    """Convert (elevation, compass azimuth) to a unit vector in the local
    ENU frame, pointing FROM a surface point TOWARD the sun -- i.e. the
    direction a shadow ray is cast in shadow.py, and the vector dotted with
    a face normal in irradiance.py to get the cosine of the incidence angle.

    East   = cos(elevation) * sin(azimuth)
    North  = cos(elevation) * cos(azimuth)
    Up     = sin(elevation)
    """
    el_rad = np.radians(elevation_deg)
    az_rad = np.radians(azimuth_deg)
    east = np.cos(el_rad) * np.sin(az_rad)
    north = np.cos(el_rad) * np.cos(az_rad)
    up = np.sin(el_rad)
    return np.stack([east, north, up], axis=1).astype(np.float32)
