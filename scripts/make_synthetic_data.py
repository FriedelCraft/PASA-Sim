"""
scripts/make_synthetic_data.py

Generates a small synthetic "city" (a handful of flat-roofed box buildings
of varying height, laid out so some overshadow their shorter neighbours)
plus matching synthetic AOD data in all four supported formats. This is
NOT real Civil Lines data -- it exists purely so PASA-Sim can be smoke-
tested end-to-end (both simulation modes, all AOD loader formats, analysis,
plotting, report generation) without needing the real GLB / MAIAC files.

Run: python scripts/make_synthetic_data.py --out-dir data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

RNG = np.random.default_rng(42)


def make_city_glb(out_path: Path, lat0: float, lon0: float) -> None:
    """A small grid of flat-roofed rectangular buildings with varied
    height, spaced so taller buildings cast shadows on shorter neighbours
    at low sun angles -- enough geometric variety to exercise both the
    back-face culling and the ray-traced occlusion path in shadow.py.
    """
    scene = trimesh.Scene()
    grid_positions = [(gx, gy) for gx in range(4) for gy in range(3)]
    heights = [22, 9, 15, 30, 8, 18, 12, 25, 10, 16, 20, 7]
    spacing = 18.0
    footprint = 12.0

    for i, ((gx, gy), height) in enumerate(zip(grid_positions, heights)):
        w = footprint + RNG.uniform(-2, 2)
        d = footprint + RNG.uniform(-2, 2)
        box = trimesh.creation.box(extents=(w, d, height))
        tf = np.eye(4)
        tf[:3, 3] = [gx * spacing, gy * spacing, height / 2.0]
        name = f"building_{i:02d}"
        scene.add_geometry(box, node_name=name, geom_name=name, transform=tf)

    scene.export(str(out_path))
    print(f"Wrote synthetic city GLB: {out_path} ({len(grid_positions)} buildings)")
    print(f"  Suggested config.yaml location: latitude={lat0}, longitude={lon0}")


def make_aod_csv(out_path: Path, times: pd.DatetimeIndex, spatially_varying: bool) -> None:
    """Synthetic MAIAC-like AOD time series: a smooth seasonal-ish base
    plus day-to-day noise plus occasional short data gaps, sampled at a
    twice-daily cadence (mimicking real satellite overpass frequency) so
    the loader's temporal interpolation path is actually exercised.
    """
    overpass_times = times[::12]  # roughly twice a day at hourly sim resolution
    n = len(overpass_times)
    base = 0.35 + 0.15 * np.sin(np.linspace(0, 3 * np.pi, n))
    noise = RNG.normal(0, 0.05, n)
    aod = np.clip(base + noise, 0.05, None)

    # punch a short gap to exercise max_gap_hours / fallback handling
    gap_start = n // 3
    aod[gap_start:gap_start + 2] = np.nan

    if spatially_varying:
        rows = []
        pixel_offsets = [(-300.0, -300.0), (300.0, -300.0), (0.0, 400.0)]
        for dx, dy in pixel_offsets:
            pixel_aod = aod * RNG.uniform(0.85, 1.15)
            for t, a in zip(overpass_times, pixel_aod):
                rows.append({"timestamp": t.isoformat(), "aod_550nm": a, "lat": 28.6752 + dy / 111_000, "lon": 77.2210 + dx / 111_000})
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame({"timestamp": overpass_times.astype(str), "aod_550nm": aod})

    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic AOD CSV: {out_path} ({len(df)} rows, spatially_varying={spatially_varying})")


def make_aod_numpy(out_path: Path, times: pd.DatetimeIndex) -> None:
    overpass_times = times[::12]
    n = len(overpass_times)
    aod = np.clip(0.4 + 0.1 * RNG.normal(size=n), 0.05, None).astype(np.float32)
    np.savez(out_path, timestamps=overpass_times.astype("int64") // 10**9, aod_550nm=aod)
    print(f"Wrote synthetic AOD NumPy: {out_path} ({n} timestamps)")


def make_aod_geotiff(out_path: Path, times: pd.DatetimeIndex, lat0: float, lon0: float) -> None:
    import rasterio
    from rasterio.transform import from_origin

    overpass_times = times[::24]  # daily bands to keep the file small
    n = len(overpass_times)
    height, width = 6, 6
    pixel_deg = 0.01
    transform = from_origin(lon0 - width * pixel_deg / 2, lat0 + height * pixel_deg / 2, pixel_deg, pixel_deg)

    data = np.clip(0.35 + 0.1 * RNG.normal(size=(n, height, width)), 0.05, None).astype(np.float32)

    with rasterio.open(
        out_path, "w", driver="GTiff", height=height, width=width, count=n,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        for band_idx in range(n):
            dst.write(data[band_idx], band_idx + 1)
        dst.update_tags(band_timestamps=",".join(t.isoformat() for t in overpass_times))
    print(f"Wrote synthetic AOD GeoTIFF: {out_path} ({n} bands, {height}x{width} px)")


def make_aod_netcdf(out_path: Path, times: pd.DatetimeIndex, lat0: float, lon0: float) -> None:
    import xarray as xr

    overpass_times = times[::12]
    n = len(overpass_times)
    lats = lat0 + np.linspace(-0.03, 0.03, 5)
    lons = lon0 + np.linspace(-0.03, 0.03, 5)
    data = np.clip(0.35 + 0.12 * RNG.normal(size=(n, 5, 5)), 0.05, None).astype(np.float32)

    ds = xr.Dataset(
        {"aod_550nm": (("time", "lat", "lon"), data)},
        coords={"time": overpass_times.tz_localize(None), "lat": lats, "lon": lons},
    )
    ds.to_netcdf(out_path)
    print(f"Wrote synthetic AOD NetCDF: {out_path} ({n} timestamps, 5x5 grid)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--lat", type=float, default=28.6752)
    parser.add_argument("--lon", type=float, default=77.2210)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--start", type=str, default=None, help="ISO date, e.g. 2024-01-10 (default: Jan 8 of --year)")
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = args.start or f"{args.year}-01-08"
    times = pd.date_range(start, periods=args.days * 24, freq="1h", tz="Asia/Kolkata")

    make_city_glb(args.out_dir / "synthetic_city.glb", args.lat, args.lon)
    make_aod_csv(args.out_dir / "synthetic_aod_single.csv", times, spatially_varying=False)
    make_aod_csv(args.out_dir / "synthetic_aod_multi_pixel.csv", times, spatially_varying=True)
    make_aod_numpy(args.out_dir / "synthetic_aod.npz", times)
    make_aod_geotiff(args.out_dir / "synthetic_aod.tif", times, args.lat, args.lon)
    make_aod_netcdf(args.out_dir / "synthetic_aod.nc", times, args.lat, args.lon)


if __name__ == "__main__":
    main()
