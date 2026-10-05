"""
io/aod_loaders.py

Pluggable AOD (Aerosol Optical Depth) file readers. Every format-specific
loader below has exactly one job: parse a file into the common
``AODDataset`` representation (a set of "pixels" -- one for spatially
uniform data, many for spatially varying data -- each with its own
irregular-cadence observation time series). Everything downstream of that
(temporal interpolation to the simulation's hourly index, gap-filling,
matching city-model faces to the nearest pixel) is format-agnostic and
lives in models/aod_beer_lambert.py, so it is written exactly once no
matter how many file formats are supported.

Adding a new format later means: subclass BaseAODLoader, decorate with
@register_loader("your_format"), map its file extension(s) in
`_EXTENSION_TO_FORMAT`, and import the new module from io/__init__.py.

Note on spatial scope: for GeoTIFF/NetCDF grids, this module loads the
*entire* grid found in the file into `AODDataset` (one "pixel" per grid
cell). It does not crop to the city's footprint -- that responsibility
belongs to the file the user supplies (pre-clip your AOD raster to
roughly the AOI before pointing PASA-Sim at it) or to
models.aod_beer_lambert, which discards any pixel not nearest to at least
one simulated face. Pointing this at an uncropped, continental-scale
raster will load the whole thing into memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import AODIngestConfig
from ..constants import AOD_PHYSICAL_MAX, AOD_PHYSICAL_MIN
from ..utilities import get_logger

logger = get_logger()

_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".npz": "numpy",
    ".npy": "numpy",
}


@dataclass
class AODDataset:
    """Common output of every AOD loader: a set of "pixels", each with its
    own (possibly irregular-cadence, possibly gappy) observation time
    series. For spatially-uniform sources there is exactly one pixel; its
    lat/lon are NaN (it doesn't correspond to a specific point and is
    intended to apply uniformly across the whole city).
    """

    times: pd.DatetimeIndex  # (n_obs,) union of all observation timestamps, tz-aware, sorted
    values: np.ndarray  # (n_pixels, n_obs) float64, NaN where a pixel has no observation at that time
    lat: np.ndarray  # (n_pixels,) float64, NaN for the spatially-uniform case
    lon: np.ndarray  # (n_pixels,) float64
    spatially_varying: bool
    source_path: str

    @property
    def n_pixels(self) -> int:
        return int(self.values.shape[0])


_LOADER_REGISTRY: dict[str, "type[BaseAODLoader]"] = {}


def register_loader(key: str) -> Callable[[type], type]:
    def _decorator(cls: type["BaseAODLoader"]) -> type["BaseAODLoader"]:
        _LOADER_REGISTRY[key] = cls
        return cls

    return _decorator


def infer_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in _EXTENSION_TO_FORMAT:
        raise ValueError(
            f"Cannot infer AOD file format from extension {suffix!r} ({path}); "
            f"set pollution.aod.format explicitly to one of {sorted(set(_EXTENSION_TO_FORMAT.values()))}"
        )
    return _EXTENSION_TO_FORMAT[suffix]


def load_aod(cfg: AODIngestConfig) -> AODDataset:
    """Factory entry point: load `cfg.path` using the loader registered
    for `cfg.format` (or the format inferred from the file extension).
    """
    if cfg.path is None:
        raise ValueError("pollution.aod.path is not set")
    fmt = cfg.format or infer_format(cfg.path)
    if fmt not in _LOADER_REGISTRY:
        raise KeyError(f"No AOD loader registered for format {fmt!r}. Known formats: {sorted(_LOADER_REGISTRY)}")
    loader_cls = _LOADER_REGISTRY[fmt]
    logger.info(f"Loading AOD data ({fmt}): {cfg.path}")
    dataset = loader_cls(cfg).load()
    logger.info(
        f"AOD dataset loaded: {dataset.n_pixels} pixel(s), {len(dataset.times)} observation timestamp(s), "
        f"spatially_varying={dataset.spatially_varying}"
    )
    return dataset


class BaseAODLoader(ABC):
    def __init__(self, cfg: AODIngestConfig):
        self.cfg = cfg

    @abstractmethod
    def load(self) -> AODDataset: ...

    def _validate_values(self, values: np.ndarray) -> np.ndarray:
        """Flag physically implausible AOD readings (sensor artifacts,
        fill values that slipped through) as NaN so they're handled by the
        exact same gap-filling path as ordinary missing observations,
        rather than silently trusted.
        """
        values = values.astype(np.float64)
        bad = (values < AOD_PHYSICAL_MIN) | (values > AOD_PHYSICAL_MAX)
        n_bad = int(np.nansum(bad))
        if n_bad:
            logger.warning(
                f"{n_bad} AOD reading(s) outside the physically plausible range "
                f"[{AOD_PHYSICAL_MIN}, {AOD_PHYSICAL_MAX}]; treating as missing data"
            )
        return np.where(bad, np.nan, values)

    def _parse_times(self, raw_times: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Parse timestamps and ensure they end up timezone-aware. Naive
        timestamps are assumed to be in `cfg.source_timezone` (UTC by
        default, matching how satellite AOD products are normally
        distributed) and localized accordingly; already-aware timestamps
        are left as given.
        """
        parsed = pd.DatetimeIndex(pd.to_datetime(raw_times))
        if parsed.tz is None:
            parsed = parsed.tz_localize(self.cfg.source_timezone)
        return parsed


@register_loader("csv")
class CSVAODLoader(BaseAODLoader):
    """Expects at minimum a timestamp column and a value column. If
    `spatially_varying` is set and lat/lon columns are present, rows are
    grouped by unique (lat, lon) pairs into separate pixels; otherwise all
    rows are treated as one spatially-uniform series.
    """

    def load(self) -> AODDataset:
        path = Path(self.cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"AOD CSV not found: {path}")
        df = pd.read_csv(path)
        for col in (self.cfg.time_column, self.cfg.value_column):
            if col not in df.columns:
                raise ValueError(f"AOD CSV {path} is missing required column {col!r}; found columns {list(df.columns)}")

        times = self._parse_times(df[self.cfg.time_column])
        has_latlon = self.cfg.lat_column in df.columns and self.cfg.lon_column in df.columns

        if self.cfg.spatially_varying and has_latlon:
            work = df[[self.cfg.value_column]].copy()
            work["_t"] = times
            work["_lat"] = df[self.cfg.lat_column]
            work["_lon"] = df[self.cfg.lon_column]
            groups = work.groupby(["_lat", "_lon"], sort=True)

            all_times = pd.DatetimeIndex(sorted(times.unique()))
            n_pixels = groups.ngroups
            values = np.full((n_pixels, len(all_times)), np.nan, dtype=np.float64)
            lat = np.zeros(n_pixels, dtype=np.float64)
            lon = np.zeros(n_pixels, dtype=np.float64)
            for i, ((glat, glon), gdf) in enumerate(groups):
                lat[i], lon[i] = glat, glon
                idx = all_times.get_indexer(gdf["_t"])
                values[i, idx] = gdf[self.cfg.value_column].to_numpy(dtype=np.float64)

            return AODDataset(
                times=all_times,
                values=self._validate_values(values),
                lat=lat, lon=lon,
                spatially_varying=True,
                source_path=str(path),
            )

        if self.cfg.spatially_varying and not has_latlon:
            logger.warning(
                f"pollution.aod.spatially_varying=True but columns "
                f"{self.cfg.lat_column!r}/{self.cfg.lon_column!r} not found in {path}; "
                "falling back to a single spatially-uniform series"
            )

        order = np.argsort(times.values)
        values = self._validate_values(df[self.cfg.value_column].to_numpy(dtype=np.float64)[order])[None, :]
        return AODDataset(
            times=pd.DatetimeIndex(times[order]),
            values=values,
            lat=np.array([np.nan]), lon=np.array([np.nan]),
            spatially_varying=False,
            source_path=str(path),
        )


@register_loader("numpy")
class NumPyAODLoader(BaseAODLoader):
    """Expects an .npz with at minimum: `timestamps` (int64 Unix seconds,
    (n_obs,)) and a value array named `cfg.value_column` (defaults to
    "aod_550nm"), shape (n_obs,) for a uniform series or (n_pixels, n_obs)
    for a spatially varying one. Optional `lat`/`lon` arrays, (n_pixels,),
    are required if the value array is 2D.
    """

    def load(self) -> AODDataset:
        path = Path(self.cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"AOD NumPy file not found: {path}")
        with np.load(path) as npz:
            if "timestamps" not in npz:
                raise ValueError(f"{path} is missing required key 'timestamps'")
            if self.cfg.value_column not in npz:
                raise ValueError(f"{path} is missing required key {self.cfg.value_column!r}; found keys {list(npz.keys())}")
            timestamps = npz["timestamps"]
            values = np.asarray(npz[self.cfg.value_column], dtype=np.float64)
            lat = np.asarray(npz["lat"], dtype=np.float64) if "lat" in npz else None
            lon = np.asarray(npz["lon"], dtype=np.float64) if "lon" in npz else None

        times = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True))

        if values.ndim == 1:
            return AODDataset(
                times=times, values=values[None, :],
                lat=np.array([np.nan]), lon=np.array([np.nan]),
                spatially_varying=False, source_path=str(path),
            )

        if lat is None or lon is None:
            raise ValueError(f"{path}: 2D value array requires 'lat' and 'lon' keys for spatially-varying data")
        return AODDataset(
            times=times, values=self._validate_values(values),
            lat=lat, lon=lon,
            spatially_varying=bool(self.cfg.spatially_varying and values.shape[0] > 1),
            source_path=str(path),
        )


@register_loader("geotiff")
class GeoTIFFAODLoader(BaseAODLoader):
    """Expects one band per timestamp. Band timestamps are read from a
    `band_timestamps` dataset tag (comma-separated ISO8601 strings, as
    written by scripts/make_synthetic_data.py) if present; otherwise the
    loader uses per-band tags `TIFFTAG_DATETIME` / `DateTime`, or daily
    timestamps beginning at an explicitly configured `start_date`.
    Missing dates raise an error rather than silently assuming a year.

    Nodata is masked before scaling. An explicit scale_factor wins over
    raster metadata; unscaled integer data defaults to MAIAC's 0.001,
    while floating-point data is treated as physical AOD.
    """

    def load(self) -> AODDataset:
        import rasterio

        path = Path(self.cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"AOD GeoTIFF not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(masked=True).astype(np.float64).filled(np.nan)
            for band, dtype in enumerate(src.dtypes):
                metadata_scale = src.scales[band]
                scale = self.cfg.scale_factor
                if scale is None:
                    scale = metadata_scale if metadata_scale not in (None, 0, 1) else (
                        0.001 if np.issubdtype(np.dtype(dtype), np.integer) else 1.0
                    )
                data[band] = data[band] * scale + src.offsets[band]
            n_bands, height, width = data.shape

            band_ts_tag = src.tags().get("band_timestamps")
            if band_ts_tag:
                times = self._parse_times(pd.Index(band_ts_tag.split(",")))
            else:
                per_band = []
                for b in range(1, n_bands + 1):
                    tag = src.tags(b).get("TIFFTAG_DATETIME") or src.tags(b).get("DateTime")
                    per_band.append(tag)
                if all(per_band):
                    times = self._parse_times(pd.Index(per_band))
                elif self.cfg.start_date is not None:
                    logger.warning("No band timestamps: using configured daily start_date=%s", self.cfg.start_date)
                    times = self._parse_times(pd.date_range(self.cfg.start_date, periods=n_bands, freq="D"))
                else:
                    raise ValueError(f"{path}: missing band timestamps; set pollution.aod.start_date explicitly")
            if len(times) != n_bands:
                raise ValueError(f"{path}: timestamp count must match the number of bands")

            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            lon, lat = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
            if src.crs is None:
                raise ValueError(f"{path}: a CRS is required to locate AOD pixels")
            if src.crs != rasterio.crs.CRS.from_epsg(4326):
                from rasterio.warp import transform
                lon, lat = transform(src.crs, "EPSG:4326", lon, lat)
            lat = np.asarray(lat, dtype=np.float64)
            lon = np.asarray(lon, dtype=np.float64)
            values = data.reshape(n_bands, height * width).T  # (n_pixels, n_bands)

        return AODDataset(
            times=times,
            values=self._validate_values(values),
            lat=lat, lon=lon,
            spatially_varying=bool(self.cfg.spatially_varying),
            source_path=str(path),
        )


@register_loader("netcdf")
class NetCDFAODLoader(BaseAODLoader):
    """Expects a `time` dimension, `lat`/`lon` (or `latitude`/`longitude`)
    coordinate variables, and a data variable named `cfg.value_column`
    with dims (time, lat, lon) -- the standard layout for MODIS/CAMS/
    MERRA-2-style gridded reanalysis or satellite products distributed as
    NetCDF.
    """

    def load(self) -> AODDataset:
        import xarray as xr

        path = Path(self.cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"AOD NetCDF not found: {path}")

        with xr.open_dataset(path) as ds:
            if self.cfg.value_column not in ds.data_vars:
                raise ValueError(
                    f"{path} is missing data variable {self.cfg.value_column!r}; "
                    f"found variables {list(ds.data_vars)}"
                )
            lat_name = "lat" if "lat" in ds.coords else "latitude"
            lon_name = "lon" if "lon" in ds.coords else "longitude"
            if lat_name not in ds.coords or lon_name not in ds.coords or "time" not in ds.coords:
                raise ValueError(f"{path} must have time/lat/lon coordinates; found {list(ds.coords)}")

            da = ds[self.cfg.value_column].transpose("time", lat_name, lon_name)
            times = self._parse_times(pd.DatetimeIndex(da["time"].values))
            lat_grid, lon_grid = np.meshgrid(da[lat_name].values, da[lon_name].values, indexing="ij")
            values = da.values.reshape(len(times), -1).T  # (n_pixels, n_obs)

        return AODDataset(
            times=times,
            values=self._validate_values(values),
            lat=lat_grid.ravel().astype(np.float64),
            lon=lon_grid.ravel().astype(np.float64),
            spatially_varying=bool(self.cfg.spatially_varying),
            source_path=str(path),
        )
