# PASA-Sim -- Pollution-Aware Solar Assessment

A single simulation framework that runs either a **perfect clear-sky** or a
**MODIS-MAIAC pollution-aware** direct-beam solar irradiance simulation over
a 3D city model, selected with one config key:

```yaml
run:
  simulation_mode: clear      # or: pollution
```

Built for the ISRO PS-I pollution-aware solar potential assessment work.
Everything is direct-beam only (DNI x cos(incidence) x shadow-visibility x
atmospheric transmission), matching the scope of the pre-upgrade baseline
pipeline this replaces, so clear-vs-pollution comparisons stay physically
apples-to-apples.

## Install

```bash
pip install -e .
# Strongly recommended for realistic city sizes -- falls back automatically
# to a slower pure-NumPy ray intersector if this isn't installed:
pip install embreex
```

Requires Python >= 3.10. See `pyproject.toml` for the full dependency list
(pinned to tested versions in `requirements.txt`).

## Quick start

```bash
# 1. Generate a small synthetic city + AOD data to sanity-check your setup
#    (NOT real data -- purely for smoke-testing the pipeline):
python scripts/make_synthetic_data.py --out-dir data/

# 2. Point config.yaml at your real GLB + MAIAC AOD file, then:
pasa simulate --config config.yaml

# 3. Run the same config with run.simulation_mode flipped to get the other run,
#    then compare:
pasa analyze outputs/clear_run.npz outputs/pollution_run.npz \
    --glb-path data/your_city.glb --output-dir report/
```

`pasa analyze` (equivalently `python -m pasa.analysis ...`, or
`python src/pasa/analysis.py ...` if you prefer running the file directly)
writes `summary.csv`, `summary.json`, `report.pdf`, and every figure (PNG +
one interactive HTML city view) into `--output-dir`.

## Architecture

```
src/pasa/
  constants.py     physical/geodetic constants + the framework's closed vocabulary
  config.py        dataclass config schema, YAML loading, validation
  utilities.py     logging, timing, config hashing, ENU<->lat/lon, chunk_ranges
  geometry.py      GLB -> per-face normals/areas/tilt/azimuth/roof-wall classification
  solar.py         pvlib sun position + Ineichen clear-sky DNI + ENU sun vectors
  models/          pluggable attenuation models (BaseAttenuationModel registry)
    base.py          the interface + registry every model implements
    clear_sky.py     ClearSkyModel: transmission == 1.0 always
    aod_beer_lambert.py   AODAttenuationModel: Beer-Lambert using MAIAC AOD
  io/
    mesh_io.py       GLB -> world-space triangles with per-building identity
    aod_loaders.py   CSV/GeoTIFF/NetCDF/NumPy -> common AODDataset
    npz_io.py        the simulation-result NPZ schema, read + write
  shadow.py        Embree (or NumPy-fallback) ray-traced sun visibility
  irradiance.py    combines geometry+solar+shadow+attenuation for one hour-chunk
  checkpoint.py    generic checkpoint/resume for a long chunked computation
  simulation.py    orchestrates the full run (the only file that reads simulation_mode)
  analysis.py      loads 1-2 NPZs, computes the full statistics suite (also standalone-runnable)
  plotting.py      all figures (static PNG + interactive Plotly HTML)
  report.py        assembles summary.csv / summary.json / report.pdf
  cli.py           `pasa simulate` / `pasa analyze`
```

**Why `models.py`/`io.py` became packages, not files:** the whole point of
"different attenuation models can easily be added later" / "future AOD
formats" is a registry you extend by adding a file, not editing existing
ones. `models/__init__.py` and `io/aod_loaders.py`'s `@register_*`
decorators are that registry.

**Why the physics is direct-beam only:** the baseline pipeline this
replaces never had a diffuse-sky (DHI) or ground-reflected component. Adding
one now would change what "pollution vs clear" means to compare, so it was
kept out of scope -- worth flagging explicitly as a limitation in a
Methodology section, and a natural next extension if you want GHI-based
results later (it would slot into `irradiance.py` without touching the
attenuation-model layer at all).

**Performance:** before any ray is cast, faces are culled by
`cos(incidence) <= 0` (self-shadowed) and hours by `sun elevation <= 0`
(night); only genuine candidates are batched into one vectorized
`intersects_any()` Embree call per hour-chunk. Multiprocessing (`n_workers >
1`) splits hour-chunks across worker processes via `ProcessPoolExecutor`,
each building its own geometry/ray-backend/AOD-interpolation once (Embree
intersectors and file handles aren't picklable, so this is a deliberate,
one-time-per-worker cost). `executor.map` preserves submission order in its
results even though workers finish out of order, which is what makes
checkpointing correct without extra bookkeeping.

**Storage scaling:** `hourly_irradiance_w_m2` and `hourly_shadow_mask` are
the only arrays that scale with `n_faces x n_hours` -- at 100,000 faces x
8,784 hours (a leap year) that's ~3.5 GB (float32) and ~0.9 GB
(bool) *before* compression. Compression helps a lot (long runs of
zero/False at night compress very well), but if you're memory- or disk-
constrained, `output.save_hourly_irradiance` / `output.save_hourly_shadow_mask`
in config.yaml can be set to `false` to keep only the always-tiny annual
totals and solar/AOD time series.

## The pollution model, and a caveat worth knowing before you write the Methodology section

```
transmission(t) = exp(-AOD(t) * airmass(t))          # Beer-Lambert
attenuated_irradiance = DNI_clear(t) * transmission(t) * cos(incidence) * visible
```

`airmass(t)` is the **relative** (geometric) optical air mass (Kasten-Young
1989 by default), not the pressure-corrected absolute air mass -- correct
because AOD is a directly-retrieved column quantity, not a surface-
pressure-scaled Rayleigh term; pressure correction belongs to Rayleigh
optical depth specifically. This matches standard practice for slant-path
aerosol corrections (e.g. AERONET-style processing).

**The caveat:** `DNI_clear` already comes from the Ineichen model driven by
a *climatological* Linke turbidity (`solar.linke_turbidity_source:
"climatology"`, pvlib's bundled dataset), which itself represents a
typical long-term aerosol + water-vapor load. Multiplying by a second,
MAIAC-derived aerosol transmission term is therefore a first-order
approximation of the *excess* attenuation from a specific pollution episode
relative to that climatology -- not a from-scratch radiative-transfer
solve. There's a theoretical risk of partially double-counting "typical"
background aerosol content. If a reviewer might push on this, set
`solar.linke_turbidity_source: "fixed"` with a low, near-Rayleigh
`fixed_linke_turbidity` (e.g. 1.5-2.0) so the clear-sky baseline carries
minimal built-in aerosol assumption, leaving (nearly) all aerosol
attenuation to the explicit AOD term -- a stricter decomposition you can
describe and defend directly.

Other assumptions worth stating explicitly in a paper: solar position/DNI
is computed once at a single AOI reference point (`location:` in the
config), not per-face -- standard and negligible-error for a district-scale
AOI, but would need revisiting for a much larger domain. Spatially-varying
AOD assigns each face to its *nearest* AOD pixel (nearest-neighbor, not
bilinear interpolation).

## NPZ output schema

One compressed `.npz` per run (`io/npz_io.py`). Per-face arrays are indexed
0..n_faces-1 in GLB scene-node order; per-hour arrays 0..n_hours-1 in
chronological order; excluded faces (undersides/slivers) are `NaN` in
`annual_irradiance_wh_m2` / `hourly_irradiance_w_m2` and must be filtered
out via `is_roof | is_wall` before use (this is exactly what
`analysis.py` does everywhere). Spatially-varying AOD is stored
deduplicated: one transmission/value series per *touched* AOD pixel
("group"), plus a `(n_faces,)` `aod_group_id` mapping each face to its
group -- not a wasteful `(n_faces, n_hours)` array full of duplicate rows.
See the `SimulationResult` dataclass in `io/npz_io.py` for the complete,
authoritative field list; `metadata_json` embeds a full config snapshot,
AOD statistics, runtime, git commit (best-effort), and schema version for
reproducibility.

## Interactive 3D view: Plotly, not PyVista

PyVista's off-screen rendering needs a working VTK/EGL context, which isn't
guaranteed on a headless machine. Plotly's `Mesh3d` produces a
self-contained HTML file (`--glb-path` enables true solid-mesh rendering by
reloading the original GLB's triangles; without it, figures fall back to a
centroid point-cloud). If you've confirmed PyVista off-screen rendering
works on your target machine and want that path instead, it would slot into
`plotting.py` alongside `plot_city_mesh_interactive` without touching
anything else.

## Testing

```bash
pytest tests/ -v                          # fast unit tests (pure math, config, NPZ round-trip)
python scripts/make_synthetic_data.py     # generates data/ fixtures used for manual smoke-testing
```

The synthetic city/AOD data is **not real Civil Lines data** -- it exists
purely to exercise every code path (both simulation modes, all four AOD
formats, checkpoint/resume, multiprocessing, the full analysis/plotting/
report pipeline) without your real GLB/MAIAC files on hand.
