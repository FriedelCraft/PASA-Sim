# Method and study scope

The Civil Lines study connects building footprints and height estimates to a Level of Detail 1 city mesh, then evaluates direct sunlight on classified roof and wall triangles.

1. Acquire and clip building footprints to the area of interest.
2. Assign building heights. The final available height extraction script samples the 2023 Google Open Buildings temporal height collection through Earth Engine, using a 4 m download grid, a median of positive samples, and a minimum of three valid pixels. Fallbacks and clipping are applied in the script.
3. Extrude footprints and generate a merged GLB. The final city model uses local metre coordinates with Z up; a separate origin JSON records the projected origin.
4. Compute hourly solar position and Ineichen clear-sky direct normal irradiance with pvlib at the configured Delhi reference location.
5. For pollution mode, match faces to AOD groups and interpolate their observations onto the hourly time axis. Gap handling and fallback values are configured in YAML.
6. Apply Beer–Lambert transmission `T = exp(-AOD * relative_airmass)` with Kasten–Young relative air mass.
7. Ray trace toward the sun from each front-facing surface centroid. Obstruction by the city mesh removes direct-beam illumination.
8. Calculate `I_face = DNI_clear * T * max(dot(normal, sun_direction), 0) * visibility` in W/m², and accumulate hourly samples in Wh/m².

Only a one-hour frequency is supported by the energy accumulator. The organized copy rejects other frequencies instead of returning incorrectly integrated energy.

The available study uses 8,760 hourly samples for 2025. It models direct-beam incident energy; diffuse sky irradiance, ground reflection, electrical conversion, module characteristics, and external validation measurements are outside this implementation. The turbidity climatology in the clear-sky baseline already represents atmospheric effects. Applying a further AOD attenuation term should be interpreted within that baseline assumption.

Triangle classifications and totals are sensitive to geometry, face-area filtering, and tilt thresholds. The final mesh does not retain footprint identifiers. Tile nodes in the older 3D Tiles export also represent tiles rather than individual buildings.
