"""Regression checks for ingestion and configuration fixes in the organized copy."""
from pathlib import Path

import numpy as np
import pytest

from pasa.analysis import _validate_comparison_alignment
from pasa.config import AODIngestConfig, ConfigError, PASAConfig
from pasa.io.aod_loaders import GeoTIFFAODLoader
from test_core_math import TestNpzRoundTrip as _NpzFixtures


def write_raster(path, data, *, timestamps=True, crs="EPSG:4326", scale=None):
    rasterio = pytest.importorskip("rasterio", exc_type=ImportError)
    from rasterio.transform import from_origin
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=2, count=1,
        dtype=data.dtype, transform=from_origin(77, 29, 0.01, 0.01),
        crs=crs, nodata=-9999,
    ) as dst:
        dst.write(data.reshape(1, 1, 2))
        if timestamps:
            dst.update_tags(band_timestamps="2024-01-08T00:00:00+00:00")
        if scale is not None:
            dst.scales = (scale,)


def test_float_aod_is_not_rescaled_and_nodata_is_masked(tmp_path):
    path = tmp_path / "aod.tif"
    write_raster(path, np.array([0.35, -9999], dtype="float32"))
    data = GeoTIFFAODLoader(AODIngestConfig(path=str(path))).load()
    assert data.values[0, 0] == pytest.approx(0.35)
    assert np.isnan(data.values[1, 0])


@pytest.mark.parametrize("metadata_scale,override,expected", [(None, None, 0.35), (0.002, None, 0.7), (0.002, 0.001, 0.35)])
def test_integer_and_explicit_aod_scales(tmp_path, metadata_scale, override, expected):
    path = tmp_path / "aod.tif"
    write_raster(path, np.array([350, -9999], dtype="int16"), scale=metadata_scale)
    data = GeoTIFFAODLoader(AODIngestConfig(path=str(path), scale_factor=override)).load()
    assert data.values[0, 0] == pytest.approx(expected)
    assert np.isnan(data.values[1, 0])


def test_missing_dates_require_explicit_start(tmp_path):
    path = tmp_path / "aod.tif"
    write_raster(path, np.array([0.35, 0.4], dtype="float32"), timestamps=False)
    with pytest.raises(ValueError, match="start_date"):
        GeoTIFFAODLoader(AODIngestConfig(path=str(path))).load()
    data = GeoTIFFAODLoader(AODIngestConfig(path=str(path), start_date="2023-02-01")).load()
    assert data.times[0].isoformat() == "2023-02-01T00:00:00+00:00"


def test_band_timestamp_count_is_checked(tmp_path):
    path = tmp_path / "aod.tif"
    write_raster(path, np.array([0.35, 0.4], dtype="float32"))
    import rasterio
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(band_timestamps="2024-01-01,2024-01-02")
    with pytest.raises(ValueError, match="timestamp count"):
        GeoTIFFAODLoader(AODIngestConfig(path=str(path))).load()


def test_projected_raster_pixels_are_converted_to_lon_lat(tmp_path):
    path = tmp_path / "aod.tif"
    write_raster(path, np.array([0.35, 0.4], dtype="float32"), crs="EPSG:3857")
    data = GeoTIFFAODLoader(AODIngestConfig(path=str(path))).load()
    assert np.max(np.abs(data.lon)) < 0.001
    assert np.max(np.abs(data.lat)) < 0.001


def test_unknown_raster_crs_is_rejected(tmp_path):
    path = tmp_path / "aod.tif"
    write_raster(path, np.array([0.35, 0.4], dtype="float32"), crs=None)
    with pytest.raises(ValueError, match="CRS"):
        GeoTIFFAODLoader(AODIngestConfig(path=str(path))).load()


def test_config_paths_are_relative_to_yaml_and_mode_override_precedes_validation(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    path = folder / "example.yaml"
    path.write_text(
        "run:\n  simulation_mode: clear\n  output_dir: ../outputs\n"
        "city_model:\n  glb_path: ../data/city.glb\n"
        "pollution:\n  aod:\n    path: ../data/aod.csv\n", encoding="utf-8",
    )
    cfg = PASAConfig.from_yaml(path, simulation_mode="pollution")
    assert cfg.pollution.attenuation_model == "aod_beer_lambert"
    assert Path(cfg.city_model.glb_path) == tmp_path / "data/city.glb"
    assert Path(cfg.pollution.aod.path) == tmp_path / "data/aod.csv"
    assert Path(cfg.run.output_dir) == tmp_path / "outputs"


@pytest.mark.parametrize("time", [{"freq": "30min"}, {"start": "2024-01-01"}, {"end": "2024-01-02"}])
def test_time_settings_cannot_silently_misstate_energy(time):
    with pytest.raises(ConfigError):
        PASAConfig.from_dict({"time": time})


@pytest.mark.parametrize("field", ["time_unix_s", "face_id", "centroid", "is_roof", "building_id_code"])
def test_comparison_rejects_misaligned_results(field):
    a = _NpzFixtures()._dummy_result()
    b = _NpzFixtures()._dummy_result()
    values = getattr(b, field)
    values.flat[0] = not values.flat[0] if values.dtype == bool else values.flat[0] + 1
    with pytest.raises(ValueError, match=field):
        _validate_comparison_alignment(a, b)


def test_comparison_accepts_matching_geometry_and_time():
    _validate_comparison_alignment(_NpzFixtures()._dummy_result(), _NpzFixtures()._dummy_result())
