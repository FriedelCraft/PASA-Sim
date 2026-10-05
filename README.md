# PASA-Sim: Pollution-Aware Solar Assessment

PASA-Sim estimates direct-beam solar irradiation on roofs and walls of a 3D city model, comparing an Ineichen clear-sky baseline with aerosol attenuation and ray-traced building shadows. This repository brings together the Civil Lines, Delhi internship study, the simulator, synthetic examples, city-model tools, and research documents.

The work was developed during an ISRO internship by Yash Saxena, with Manan Monga and Sameer Saran listed on the research paper. The simulation period for the Civil Lines study is 2025. Some upstream inputs and the full-city simulation NPZ files are absent from the supplied workspace. Saved research outputs are preserved separately from runnable examples.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/pasa/` | Installable simulator, analysis, plotting, and reporting package |
| `configs/` | Separate clear and pollution configurations for synthetic and Civil Lines runs |
| `tests/` | Numerical tests and regression checks |
| `scripts/` | Synthetic fixture generation and city-model preparation tools |
| `research/papers/` | Full ACRS 2026 paper, extended abstract, and earlier proposal abstract |
| `research/templates/` | Original conference submission template |
| `research/presentations/`, `research/reports/` | Earlier internship presentation and additional mentor report |
| `data/synthetic/` | Small city model and sample AOD files for a reproducible demonstration |
| `data/civil_lines/` | Final city GLB, building footprints/heights, boundary, origin, height statistics, and available AOD raster |
| `data/earlier_pipeline/` | Earlier CityJSON, 3D Tiles, zero-valued nDSM, pipeline log, and statistics |
| `data/experimental/` | Alternative merged city models; coordinate handling requires validation |
| `results/civil_lines/` | Saved full-city reports, statistics, and figures |
| `results/synthetic/` | Original synthetic annual reports and one-week demonstration |
| `archive/` | Historical generators, diagnostics, figure scripts, and incomplete pipeline fragment |
| `docs/` | Method, provenance, reproducibility notes, changes, verification, and original-file manifest |

## Install and run a small example

Python 3.10 or newer is required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\pasa.exe simulate --config configs/synthetic_clear.yaml
.\.venv\Scripts\pasa.exe simulate --config configs/synthetic_pollution.yaml
.\.venv\Scripts\pasa.exe analyze outputs/synthetic/clear.npz outputs/synthetic/pollution.npz --glb-path data/synthetic/synthetic_city.glb --output-dir outputs/synthetic/comparison
```

On Linux/macOS, use `.venv/bin/python` and `.venv/bin/pasa`. Optional Embree acceleration is installed with `pip install -e ".[fast-raytracing]"`; the standard install includes `rtree` for the triangle-ray fallback.

Configuration file paths resolve relative to the YAML file. The synthetic example covers January 8–12, 2024. Its files contain fabricated AOD observations and twelve box-shaped buildings. Use `configs/synthetic_geotiff.yaml` to exercise the raster loader. Generated outputs and virtual environments are ignored by Git.

## Civil Lines study

```powershell
pasa simulate --config configs/civil_lines_clear.yaml
pasa simulate --config configs/civil_lines_pollution.yaml
```

These are full-year, full-city jobs and require substantially more time than the synthetic example. They save annual totals without the large hourly arrays by default. Enable hourly irradiance explicitly if temporal plots are needed. See [reproducibility](docs/REPRODUCIBILITY.md) before interpreting a rerun as a reproduction of the paper.

The final merged mesh contains 126,402 triangles and was generated from 8,351 footprints. It stores the city as one scene geometry, so its simulator building identifier does **not** recover 8,351 individual buildings. Preserved reports and paper counts differ slightly in roof classification; this is documented in [data provenance](docs/DATA_PROVENANCE.md).

The simulator estimates incident direct-beam energy. It does not model diffuse irradiance, ground reflection, PV conversion efficiency, or measured electricity generation. Aerosol attenuation is applied on top of a turbidity-based baseline, so the reported difference is a model sensitivity rather than an independently calibrated aerosol-only loss.

## Research and saved results

- [Full research paper](research/papers/Pollution_Aware_Solar_Potential_ACRS2026_FINAL.docx)
- [Extended abstract](research/papers/Extended%20Abstract%20New%20SP%20R0.docx)
- [Full-city analysis report](results/civil_lines/analysis_report.pdf)
- [Mentor discussion report](results/civil_lines/ISRO_Mentor_Discussion_Report.pdf)
- [Publication figures](results/civil_lines/final_figures/)

The Word documents and saved results are copied without editing. Historical mentor materials include a tariff-based cost illustration; it should not be treated as PV revenue because incident energy has not been converted to electrical output.

## Organization and Git

This folder is self-contained: no source or configuration references files in its parent directory. All 648 supplied files are accounted for in [source_manifest.json](docs/source_manifest.json). Exact duplicate exports, ZIP copies of extracted files, Python bytecode, package installation metadata, and download caches are omitted where documented. Unique historical material is retained.

Push **this directory** as the repository root. Keep generated `outputs/`, `checkpoints/`, and `.venv/` out of commits. The available data is copied locally; no restricted missing inputs have been reconstructed or acquired. Data ownership and source attribution remain those of the original project. The original package metadata declares MIT; that declaration is retained. No license text was supplied, and no new license has been assigned to the research documents or data.

See [changes](docs/CHANGES.md) for the small code fixes made only in this copy and [verification](docs/VERIFICATION.md) for checks completed during organization.
