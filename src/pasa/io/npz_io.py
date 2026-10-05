"""
io/npz_io.py

Read/write the PASA-Sim simulation result format: a single compressed NPZ
file per run, containing every per-face geometric attribute, every
per-hour solar/DNI quantity, the (deduplicated, group-based) AOD/
transmission series, the big hourly per-face irradiance/shadow arrays
(each individually toggleable via config.OutputConfig, since they
dominate file size), annual per-face totals, and a JSON metadata blob
covering everything needed to interpret the file later without the
original config or logs on hand.

No pickling. Metadata that doesn't fit numpy's native array types
(strings, nested dicts) is serialized to a single JSON string and stored
as a 0-d numpy string array -- this keeps `allow_pickle=False` usable
everywhere, which matters for both security and cross-machine
portability (a NPZ written on one Python/numpy version should still be
loadable from another).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import NPZ_SCHEMA_VERSION


@dataclass
class SimulationResult:
    # -- per-face (n_faces,) ------------------------------------------------
    face_id: np.ndarray
    building_id_code: np.ndarray
    is_roof: np.ndarray
    is_wall: np.ndarray
    area_m2: np.ndarray
    normal: np.ndarray  # (n_faces, 3)
    centroid: np.ndarray  # (n_faces, 3)
    tilt_deg: np.ndarray
    azimuth_deg: np.ndarray
    aod_group_id: np.ndarray  # (n_faces,) int32; all zeros in clear mode
    annual_irradiance_wh_m2: np.ndarray

    # -- per-hour (n_hours,) -------------------------------------------------
    time_unix_s: np.ndarray
    solar_zenith_deg: np.ndarray
    solar_azimuth_deg: np.ndarray
    solar_elevation_deg: np.ndarray
    dni_clear_w_m2: np.ndarray
    dni_extra_w_m2: np.ndarray

    # -- per-hour-per-face (n_faces, n_hours), optional ----------------------
    hourly_irradiance_w_m2: np.ndarray | None
    hourly_shadow_mask: np.ndarray | None  # True == shadowed / no direct sun

    # -- per-hour-per-group (n_groups, n_hours), optional (pollution mode) --
    aod_group_transmission: np.ndarray | None
    aod_group_value: np.ndarray | None
    aod_group_lat: np.ndarray | None
    aod_group_lon: np.ndarray | None

    # -- lookups / metadata ---------------------------------------------------
    building_id_lookup: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_faces(self) -> int:
        return int(self.face_id.shape[0])

    @property
    def n_hours(self) -> int:
        return int(self.time_unix_s.shape[0])

    def times(self) -> pd.DatetimeIndex:
        tz = self.metadata.get("timezone", "UTC")
        return pd.to_datetime(self.time_unix_s, unit="s", utc=True).tz_convert(tz)

    def building_ids(self) -> np.ndarray:
        """Per-face original building-name strings (not the int code)."""
        lookup = np.array(self.building_id_lookup, dtype=object)
        return lookup[self.building_id_code]


def save_simulation_result(result: SimulationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_full = dict(result.metadata)
    metadata_full["npz_schema_version"] = NPZ_SCHEMA_VERSION
    metadata_full["building_id_lookup"] = result.building_id_lookup
    metadata_full["n_faces"] = result.n_faces
    metadata_full["n_hours"] = result.n_hours

    arrays: dict[str, np.ndarray] = {
        "face_id": result.face_id,
        "building_id_code": result.building_id_code,
        "is_roof": result.is_roof,
        "is_wall": result.is_wall,
        "area_m2": result.area_m2,
        "normal": result.normal,
        "centroid": result.centroid,
        "tilt_deg": result.tilt_deg,
        "azimuth_deg": result.azimuth_deg,
        "aod_group_id": result.aod_group_id,
        "annual_irradiance_wh_m2": result.annual_irradiance_wh_m2,
        "time_unix_s": result.time_unix_s,
        "solar_zenith_deg": result.solar_zenith_deg,
        "solar_azimuth_deg": result.solar_azimuth_deg,
        "solar_elevation_deg": result.solar_elevation_deg,
        "dni_clear_w_m2": result.dni_clear_w_m2,
        "dni_extra_w_m2": result.dni_extra_w_m2,
        "metadata_json": np.array(json.dumps(metadata_full, default=str)),
    }
    for key, value in (
        ("hourly_irradiance_w_m2", result.hourly_irradiance_w_m2),
        ("hourly_shadow_mask", result.hourly_shadow_mask),
        ("aod_group_transmission", result.aod_group_transmission),
        ("aod_group_value", result.aod_group_value),
        ("aod_group_lat", result.aod_group_lat),
        ("aod_group_lon", result.aod_group_lon),
    ):
        if value is not None:
            arrays[key] = value

    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    np.savez_compressed(path, **arrays)
    return path


def load_simulation_result(path: str | Path) -> SimulationResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Simulation result NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"]))
        schema_version = metadata.get("npz_schema_version")
        if schema_version != NPZ_SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written with NPZ schema version {schema_version!r}, "
                f"but this version of PASA-Sim expects {NPZ_SCHEMA_VERSION!r}. "
                "Re-run the simulation, or use a matching PASA-Sim version to read it."
            )

        def _get(key: str) -> np.ndarray | None:
            return npz[key] if key in npz.files else None

        return SimulationResult(
            face_id=npz["face_id"],
            building_id_code=npz["building_id_code"],
            is_roof=npz["is_roof"],
            is_wall=npz["is_wall"],
            area_m2=npz["area_m2"],
            normal=npz["normal"],
            centroid=npz["centroid"],
            tilt_deg=npz["tilt_deg"],
            azimuth_deg=npz["azimuth_deg"],
            aod_group_id=npz["aod_group_id"],
            annual_irradiance_wh_m2=npz["annual_irradiance_wh_m2"],
            time_unix_s=npz["time_unix_s"],
            solar_zenith_deg=npz["solar_zenith_deg"],
            solar_azimuth_deg=npz["solar_azimuth_deg"],
            solar_elevation_deg=npz["solar_elevation_deg"],
            dni_clear_w_m2=npz["dni_clear_w_m2"],
            dni_extra_w_m2=npz["dni_extra_w_m2"],
            hourly_irradiance_w_m2=_get("hourly_irradiance_w_m2"),
            hourly_shadow_mask=_get("hourly_shadow_mask"),
            aod_group_transmission=_get("aod_group_transmission"),
            aod_group_value=_get("aod_group_value"),
            aod_group_lat=_get("aod_group_lat"),
            aod_group_lon=_get("aod_group_lon"),
            building_id_lookup=list(metadata["building_id_lookup"]),
            metadata=metadata,
        )
