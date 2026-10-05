# Data and saved-result provenance

## Current study inputs

| File in `data/civil_lines/` | Meaning |
| --- | --- |
| `city.glb` | Final merged city mesh, 126,402 triangles, local metre coordinates; identical to the original City Maker and simulator final model copies |
| `city_origin.json` | Projected origin metadata, EPSG:32643; origin approximately (717175.6533, 3174555.0074) |
| `buildings_aoi.gpkg` | 8,351 clipped footprints; earlier default height attributes are not the final extracted heights |
| `CIVIL_LINES_buildings_height.gpkg` | Later 8,351-footprint height dataset; older base/roof fields coexist with the newer height field |
| `height_statistics.csv` | Final heights, approximately 3–31.38 m; mean 6.08 m |
| `CIVIL_LINES_boundary.gpkg` | Civil Lines ward boundary |
| `CIVIL_LINES_buildings.gpkg` | Separate 809-feature OSM building collection, rather than the final 8,351-footprint source |
| `mcd19a2_Optical_Depth_055_2025.tif` | Available MAIAC 550 nm AOD export: 365 bands over a 5×6 grid |

The AOD raster lacks acquisition timestamp tags. The original loader silently assumed daily bands beginning January 1, 2025 and a scale of 0.001. The new Civil Lines configuration states those assumptions explicitly. They have not been independently verified against the original satellite acquisition records. Full MAIAC quality flags and the original export workflow are unavailable.

## Historical and experimental data

`data/earlier_pipeline/` preserves the older CityJSON/3D Tiles export, nDSM, statistics, presentation provenance, and pipeline log. The available nDSM contains only zeros; it cannot demonstrate measured building heights. The older city export uses approximately 2.5–6 m default heights and differs from the final extracted-height mesh.

`data/experimental/` preserves alternative GLB reconstructions. Some are expressed in ECEF axes or recentered without conversion to East–North–Up. They are not selected by the runnable configurations. Validate geometry and coordinate conventions before using them for solar calculations. The ENU reconstruction's original sidecar is preserved as historical metadata; its simple translation note does not describe the inverse rotation required for ENU coordinates.

The supplied `aoi.py` contains duplicated source, a misplaced future import, and a dependency on a missing `.utils` module. It is preserved verbatim as `archive/incomplete_pipeline/aoi.py.txt` to distinguish it from executable tools.

## Result groups

- `results/civil_lines/` contains the full-city reports, statistics, and figure sets from the original analysis folder. Full-city `clear.npz` and `pollution.npz` files are absent.
- `results/synthetic/annual/` contains the original synthetic annual reports, despite some source labels referring to Civil Lines. Their twelve-building geometry identifies them as demos.
- `results/synthetic/week_demo/` contains the earlier 168-hour synthetic report and associated images.
- Research papers and the extended abstract are preserved byte for byte. Their claims are not rewritten to match a new run.

The paper and extended abstract report 23,234 roof faces and 79,934 wall faces (103,168 receiving faces). Saved full-city analysis reports use 23,226 roof faces and 79,934 wall faces (103,160 receiving faces): an eight-roof-face difference that remains unresolved without the original run arrays and settings.

The saved statistics report mean annual irradiation of about 650.04 kWh/m² for clear mode and 434.15 kWh/m² for pollution mode. The ratio of those means gives a 33.21% reduction; the mean of individual face percentage losses is about 22.34%. These are different summaries. Historical tariff calculations are preserved as illustrations, not validated electricity revenues.

Some upstream data was unavailable in the supplied internship workspace. This organization has not obtained or reconstructed those restricted inputs, and no local data has been uploaded by this task. Source licenses and redistribution permissions remain those of the supplied material.
