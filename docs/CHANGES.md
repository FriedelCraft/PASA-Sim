# Changes made in the organized copy

Original files outside this directory are not edited. Papers, figures, saved reports, datasets, and historical scripts are retained without modifying their content.

- Organized the simulator as the root installable package; grouped current data, historical exports, experiments, synthetic results, research, and archive material.
- Removed redundant ZIP exports and repeated source copies, bytecode, download caches, and installed package metadata. Every original file is represented in `source_manifest.json` with its initial SHA-256 hash and disposition.
- Added a root README, method/provenance/reproducibility documentation, Git ignore rules, and binary/text attributes.
- Added separate clear/pollution run names and configurations for 2025 Civil Lines and five-day synthetic examples, avoiding accidental output overwrite from the original misleading 2024 clear run name.
- Made YAML paths resolve relative to the configuration file and read configuration as UTF-8.
- Applied the CLI mode override before configuration validation, allowing a pollution override to use a configured AOD source without the earlier clear-mode mutation.
- Rejected non-hourly simulation frequencies and incomplete start/end pairs because the accumulator assumes one-hour samples.
- Fixed GeoTIFF nodata masking before scaling; floating-point physical AOD is no longer automatically multiplied by 0.001. Added an explicit scale override and per-band metadata scaling.
- Replaced the hardcoded raster year fallback with an explicit configured daily start date; checked timestamp/band count; converted projected raster coordinates to WGS84 and rejected rasters with no CRS.
- Added the missing `rtree` dependency for ray tracing without Embree and an optional city-model dependency group.
- Updated the plotting colormap calls to work with current Matplotlib versions; checked the complete synthetic report workflow.
- Added regression checks for configuration paths/mode, raster scaling/nodata/timestamps/CRS, and comparison alignment.
- Preserved the original package's MIT metadata; added no new license covering data or research documents.

The incomplete, syntactically invalid legacy AOI fragment is stored as text instead of being presented as runnable Python. It remains byte-identical to the original.
