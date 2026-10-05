# Verification completed during organization

Verified on Windows with Python 3.12. The exact installed dependency versions are recorded in `requirements-verified.txt`.

| Check | Outcome |
| --- | --- |
| Original-file integrity | All 648 supplied files matched their initial SHA-256 hashes; no originals changed |
| Python syntax | All 39 Python files in the organized source, tools, tests, and archive parsed successfully |
| Unit and regression tests | 35 passed; 8 raster checks skipped because Windows Application Control blocked the Rasterio native DLL |
| Configuration checks | All five YAML configurations validated; referenced mesh and AOD files existed |
| Synthetic clear run | Completed 120 hours on 12 buildings, 144 total faces, and 120 receiving faces |
| Synthetic pollution run | Completed the same 120 hours with spatial CSV AOD and the triangle ray backend |
| Numerical smoke checks | Receiving-face totals were finite; pollution irradiation did not exceed the matching clear baseline |
| Comparison report | Generated CSV, JSON, a readable 45-page PDF, and 38 PNG/HTML figure artifacts |
| Git ignore checks | Excluded virtual environment, generated outputs, checkpoints, caches, and installation metadata; retained the historical pipeline log |
| Git file sizes | Largest intended repository file approximately 8.85 MB |

The raster ingestion fixes have regression tests but could not be exercised against real Rasterio here because of the native-library policy block. They remain unverified in this environment. The CSV-based demonstration, ray tracing, comparison, and report generation completed successfully.

Full-year Civil Lines simulations, authenticated Earth Engine downloads, optional city-model tools, and Embree acceleration were not run. Saved research results were preserved, not regenerated. Missing full-city NPZ outputs and upstream source data prevent an exact reproduction of the paper; see `REPRODUCIBILITY.md`.

Temporary verification environments and generated demo outputs were removed from the organized directory after these checks. They can be recreated with the root README commands. `verification-details.json` records original-file dispositions and the copied source paths changed during organization.
