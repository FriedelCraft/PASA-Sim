# Reproducibility

## Runnable demonstration

The synthetic CSV configurations exercise mesh loading, solar position, atmospheric transmission, shadow tracing, NPZ output, comparison, and report generation. They cover five days on twelve box buildings. The separate GeoTIFF configuration exercises a raster that already stores physical floating-point AOD.

The checked-in synthetic fixtures can be used directly. To regenerate them deliberately:

```powershell
python scripts/make_synthetic_data.py --out-dir outputs/regenerated_synthetic --year 2024 --days 5
```

Using an output directory avoids replacing the checked-in fixtures.

## Full-city runs

The Civil Lines configurations use the final GLB and available AOD export. Default output settings save annual totals and omit hourly irradiance/shadow arrays to reduce memory and disk use. Hourly arrays are required for most temporal diagnostics. The non-Embree ray backend is suitable for a small demonstration; a full-year city run benefits from optional Embree acceleration and sufficient RAM.

These configurations are organized starting points, not an exact reconstruction of the paper run. The original full-city NPZ arrays, complete acquisition provenance, and exact settings that explain the eight-face count discrepancy are unavailable. The organization also fixes GeoTIFF ingestion; this may change rerun results where the original nodata/scaling behavior mattered.

## City-model preparation

The optional `city-model` extra installs the main dependencies used by the retained preparation scripts:

```powershell
python -m pip install -e ".[city-model]"
```

`scripts/city_model/build_city_mesh.py` expects `CIVIL_LINES_buildings_height.gpkg` in its working directory. Run it in a scratch folder containing a copy of that dataset; outputs are relative to that working directory. `extract_building_heights.py` and the Earth Engine diagnostics require authentication and a cloud project you can access; the scripts preserve the original project setting and should be configured before use. `download_osm_buildings.py` is interactive and expects `delhi_wards.kml` in its working directory. A copy is retained in `data/earlier_pipeline/`.

The 3D Tiles reconstruction tool accepts explicit input and output paths:

```powershell
python scripts/city_model/reconstruct_glb_from_3dtiles.py --help
```

Use `--enu` for a local East–North–Up frame. This operates on the older tile export and does not recreate the final extracted-height study mesh.

## Historical source

`archive/analysis/` scripts assume missing `clear.npz` and `pollution.npz` files and their original working-directory layouts. Their inclusion preserves the analysis work; they are not the quick-start entry points. `docs/original_simulator_readme.md`, `archive/config-original.yaml`, and `requirements-original.txt` preserve the original installation/configuration record. Follow the root README for the organized copy.

Package versions installed during verification are recorded in `docs/requirements-verified.txt`. Reproduce that environment with `python -m pip install -r docs/requirements-verified.txt`, then install this package with `python -m pip install -e . --no-deps`. The recorded environment was tested on Windows with Python 3.12; different platforms may need different binary wheels.
