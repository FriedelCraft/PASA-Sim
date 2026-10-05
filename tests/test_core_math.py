"""
tests/test_core_math.py

Unit tests for the pure, deterministic math/utility functions -- the
pieces cheap enough to lock down with real assertions rather than only
eyeballing simulation output. Run with:

    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pasa import utilities
from pasa.config import PASAConfig, ConfigError
from pasa.geometry import _compute_tilt_azimuth
from pasa.io.npz_io import SimulationResult, save_simulation_result, load_simulation_result


class TestUtilities:
    def test_wrap_angle_deg(self):
        assert utilities.wrap_angle_deg(370.0) == pytest.approx(10.0)
        assert utilities.wrap_angle_deg(-10.0) == pytest.approx(350.0)
        assert utilities.wrap_angle_deg(0.0) == pytest.approx(0.0)

    def test_safe_divide_avoids_div_by_zero(self):
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([1.0, 0.0, 3.0])
        result = utilities.safe_divide(num, den, fill=-1.0)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == -1.0
        assert result[2] == pytest.approx(1.0)

    def test_enu_to_latlon_zero_offset_is_identity(self):
        lat, lon = utilities.enu_to_latlon(0.0, 0.0, 28.6752, 77.2210)
        assert lat == pytest.approx(28.6752)
        assert lon == pytest.approx(77.2210)

    def test_enu_to_latlon_north_increases_latitude(self):
        lat, _ = utilities.enu_to_latlon(0.0, 1000.0, 28.0, 77.0)
        assert lat > 28.0

    def test_enu_to_latlon_east_increases_longitude(self):
        _, lon = utilities.enu_to_latlon(1000.0, 0.0, 28.0, 77.0)
        assert lon > 77.0

    def test_chunk_ranges_covers_full_range_no_gaps_no_overlap(self):
        chunks = utilities.chunk_ranges(100, 30)
        assert chunks == [(0, 30), (30, 60), (60, 90), (90, 100)]

    def test_chunk_ranges_exact_multiple(self):
        chunks = utilities.chunk_ranges(90, 30)
        assert chunks == [(0, 30), (30, 60), (60, 90)]

    def test_chunk_ranges_empty(self):
        assert utilities.chunk_ranges(0, 30) == []

    def test_hash_config_order_independent(self):
        h1 = utilities.hash_config({"a": 1, "b": {"c": 2, "d": 3}})
        h2 = utilities.hash_config({"b": {"d": 3, "c": 2}, "a": 1})
        assert h1 == h2

    def test_hash_config_sensitive_to_value_change(self):
        h1 = utilities.hash_config({"a": 1})
        h2 = utilities.hash_config({"a": 2})
        assert h1 != h2


class TestGeometryAngles:
    def test_flat_upward_face_has_zero_tilt_and_nan_azimuth(self):
        normals = np.array([[0.0, 0.0, 1.0]])
        tilt, azimuth = _compute_tilt_azimuth(normals)
        assert tilt[0] == pytest.approx(0.0)
        assert np.isnan(azimuth[0])

    def test_vertical_north_facing_wall(self):
        # ENU: North = +Y. A wall whose normal points due North should have tilt=90, azimuth=0.
        normals = np.array([[0.0, 1.0, 0.0]])
        tilt, azimuth = _compute_tilt_azimuth(normals)
        assert tilt[0] == pytest.approx(90.0)
        assert azimuth[0] == pytest.approx(0.0, abs=1e-4)

    def test_vertical_east_facing_wall(self):
        normals = np.array([[1.0, 0.0, 0.0]])
        tilt, azimuth = _compute_tilt_azimuth(normals)
        assert tilt[0] == pytest.approx(90.0)
        assert azimuth[0] == pytest.approx(90.0, abs=1e-4)

    def test_downward_face_has_180_tilt(self):
        normals = np.array([[0.0, 0.0, -1.0]])
        tilt, _ = _compute_tilt_azimuth(normals)
        assert tilt[0] == pytest.approx(180.0)

    def test_azimuth_wraps_into_0_360(self):
        # South-West-ish normal -> azimuth should land in [0, 360)
        normals = np.array([[-0.5, -0.5, 0.707]])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        _, azimuth = _compute_tilt_azimuth(normals)
        assert 0.0 <= azimuth[0] < 360.0


class TestBeerLambertBounds:
    def test_transmission_is_between_zero_and_one(self):
        aod = np.array([0.0, 0.3, 1.0, 3.0])
        airmass = np.array([1.0, 1.0, 1.0, 1.0])
        transmission = np.exp(-aod * airmass)
        assert np.all(transmission >= 0.0)
        assert np.all(transmission <= 1.0)

    def test_zero_aod_gives_full_transmission(self):
        assert np.exp(-0.0 * 2.5) == pytest.approx(1.0)

    def test_transmission_decreases_with_airmass(self):
        aod = 0.4
        t_low_airmass = np.exp(-aod * 1.0)
        t_high_airmass = np.exp(-aod * 5.0)
        assert t_high_airmass < t_low_airmass


class TestConfig:
    def test_defaults_are_valid(self):
        cfg = PASAConfig()
        assert cfg.run.simulation_mode == "clear"

    def test_clear_mode_forces_attenuation_model(self):
        cfg = PASAConfig.from_dict({
            "run": {"simulation_mode": "clear"},
            "pollution": {"attenuation_model": "aod_beer_lambert"},
        })
        assert cfg.pollution.attenuation_model == "clear"

    def test_pollution_mode_requires_aod_path(self):
        with pytest.raises(ConfigError):
            PASAConfig.from_dict({"run": {"simulation_mode": "pollution"}})

    def test_unknown_key_raises(self):
        with pytest.raises(ConfigError):
            PASAConfig.from_dict({"run": {"nonexistent_field": 1}})

    def test_bad_tilt_ordering_raises(self):
        with pytest.raises(ConfigError):
            PASAConfig.from_dict({"city_model": {"roof_tilt_threshold_deg": 150, "wall_tilt_max_deg": 100}})


class TestNpzRoundTrip:
    def _dummy_result(self, n_faces=10, n_hours=24) -> SimulationResult:
        return SimulationResult(
            face_id=np.arange(n_faces, dtype=np.int64),
            building_id_code=np.zeros(n_faces, dtype=np.int32),
            is_roof=np.ones(n_faces, dtype=bool),
            is_wall=np.zeros(n_faces, dtype=bool),
            area_m2=np.ones(n_faces, dtype=np.float32),
            normal=np.tile([0, 0, 1], (n_faces, 1)).astype(np.float32),
            centroid=np.zeros((n_faces, 3), dtype=np.float32),
            tilt_deg=np.zeros(n_faces, dtype=np.float32),
            azimuth_deg=np.full(n_faces, np.nan, dtype=np.float32),
            aod_group_id=np.zeros(n_faces, dtype=np.int32),
            annual_irradiance_wh_m2=np.ones(n_faces, dtype=np.float32) * 100.0,
            time_unix_s=np.arange(n_hours, dtype=np.int64) * 3600,
            solar_zenith_deg=np.zeros(n_hours, dtype=np.float32),
            solar_azimuth_deg=np.zeros(n_hours, dtype=np.float32),
            solar_elevation_deg=np.zeros(n_hours, dtype=np.float32),
            dni_clear_w_m2=np.zeros(n_hours, dtype=np.float32),
            dni_extra_w_m2=np.zeros(n_hours, dtype=np.float32),
            hourly_irradiance_w_m2=None,
            hourly_shadow_mask=None,
            aod_group_transmission=None,
            aod_group_value=None,
            aod_group_lat=None,
            aod_group_lon=None,
            building_id_lookup=["building_0"],
            metadata={"timezone": "UTC", "simulation_mode": "clear"},
        )

    def test_round_trip_preserves_arrays(self, tmp_path):
        result = self._dummy_result()
        path = save_simulation_result(result, tmp_path / "test.npz")
        loaded = load_simulation_result(path)
        assert np.array_equal(loaded.area_m2, result.area_m2)
        assert loaded.n_faces == result.n_faces
        assert loaded.n_hours == result.n_hours

    def test_schema_version_mismatch_is_rejected(self, tmp_path):
        result = self._dummy_result()
        path = save_simulation_result(result, tmp_path / "test.npz")
        # Corrupt the schema version to simulate a file from an incompatible future/past version
        with np.load(path) as npz:
            data = dict(npz)
        import json
        meta = json.loads(str(data["metadata_json"]))
        meta["npz_schema_version"] = "0.0.1"
        data["metadata_json"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError):
            load_simulation_result(path)
