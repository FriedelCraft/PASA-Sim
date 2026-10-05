"""
report.py

Assembles analysis.py's numbers and plotting.py's figures into the three
deliverables requested: summary.csv, summary.json, and a multi-page PDF
report (executive summary, key observations, every figure, full
statistics tables, simulation metadata). Built with matplotlib's
PdfPages so the whole plotting + report pipeline shares one dependency
and one visual style, rather than pulling in a second, heavier PDF-layout
library purely for a handful of text/table pages.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .analysis import AnalysisReport, SingleRunAnalysis
from .io.npz_io import SimulationResult
from .plotting import generate_all_figures, setup_publication_style
from .utilities import get_logger

logger = get_logger()


def _stats_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return {str(k): _stats_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _stats_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stats_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _single_to_dict(single: SingleRunAnalysis) -> dict[str, Any]:
    return {
        "source": single.source_path,
        "simulation_mode": single.simulation_mode,
        "total_energy_kwh": single.total_energy_kwh,
        "overall_stats": single.overall_stats,
        "roof_stats": single.roof_stats,
        "wall_stats": single.wall_stats,
        "orientation": single.orientation,
        "n_buildings": int(len(single.building_table)),
        "top_buildings": single.ranked_buildings["top"].to_dict(orient="records"),
        "bottom_buildings": single.ranked_buildings["bottom"].to_dict(orient="records"),
    }


def write_summary_json(report: AnalysisReport, output_path: Path) -> Path:
    payload: dict[str, Any] = {"runs": [_single_to_dict(s) for s in report.single]}
    if report.comparison is not None:
        c = report.comparison
        payload["comparison"] = {
            "total_energy_loss_kwh": c.total_energy_loss_kwh,
            "total_loss_pct": c.total_loss_pct,
            "loss_wh_m2_stats": c.loss_wh_m2_stats,
            "loss_pct_stats": c.loss_pct_stats,
            "most_affected_buildings": c.building_comparison.head(10).to_dict(orient="records"),
            "least_affected_buildings": c.building_comparison.tail(10).to_dict(orient="records"),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(_stats_to_jsonable(payload), fh, indent=2)
    return output_path


def write_summary_csv(report: AnalysisReport, output_path: Path) -> Path:
    """The primary per-building table: one row per building for the first
    (or only) run, with clear/pollution energy + loss columns appended
    when a comparison is available -- the single flat table most useful
    for a quick look in Excel/Sheets.
    """
    if report.comparison is not None:
        df = report.comparison.building_comparison.copy()
    else:
        df = report.single[0].building_table.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _generate_key_observations(report: AnalysisReport) -> list[str]:
    """A handful of natural-language bullets computed directly from the
    already-computed statistics -- every number here traces back to a real
    field in `report`, nothing is invented for narrative effect.
    """
    obs: list[str] = []
    for single in report.single:
        roof_mean = single.roof_stats["mean"]
        wall_mean = single.wall_stats["mean"]
        if not (np.isnan(roof_mean) or np.isnan(wall_mean) or wall_mean == 0):
            ratio = roof_mean / wall_mean
            obs.append(
                f"[{single.simulation_mode}] Roofs averaged {roof_mean:,.0f} Wh/m^2 vs "
                f"{wall_mean:,.0f} Wh/m^2 for walls ({ratio:.2f}x)."
            )
        obs.append(f"[{single.simulation_mode}] Total simulated-period energy yield: {single.total_energy_kwh:,.1f} kWh.")
        if single.shading_table is not None and len(single.shading_table):
            most_shaded = single.shading_table.iloc[-1]
            obs.append(
                f"[{single.simulation_mode}] Most-shaded building: {most_shaded['building_id']} "
                f"({most_shaded['shaded_fraction']*100:.0f}% of daylight hours shadowed)."
            )

    if report.comparison is not None:
        c = report.comparison
        obs.append(f"Pollution reduced total citywide energy yield by {c.total_loss_pct:.1f}% ({c.total_energy_loss_kwh:,.1f} kWh).")
        obs.append(
            f"Per-face energy loss ranged {c.loss_pct_stats['min']:.1f}%-{c.loss_pct_stats['max']:.1f}% "
            f"(median {c.loss_pct_stats['median']:.1f}%)."
        )
        worst = c.building_comparison.iloc[0]
        obs.append(f"Most pollution-affected building: {worst['building_id']} ({worst['loss_pct']:.1f}% energy lost).")
    return obs


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.05, 0.95, title, fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
    y = 0.87
    for line in lines:
        ax.text(0.05, y, line, fontsize=10, va="top", wrap=True, transform=ax.transAxes)
        y -= 0.035
        if y < 0.05:
            break
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 25) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    shown = df.head(max_rows)
    table = ax.table(
        cellText=shown.round(2).astype(str).to_numpy(),
        colLabels=shown.columns.tolist(),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    pdf.savefig(fig)
    plt.close(fig)


def _add_figure_page(pdf: PdfPages, image_path: Path, caption: str) -> None:
    import matplotlib.image as mpimg

    fig, ax = plt.subplots(figsize=(8.5, 9))
    try:
        img = mpimg.imread(image_path)
    except Exception as exc:  # noqa: BLE001 - a single bad figure must not abort the whole PDF
        logger.warning(f"Could not embed figure {image_path} in PDF: {exc}")
        plt.close(fig)
        return
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(
    report: AnalysisReport,
    results: list[SimulationResult],
    output_dir: str | Path,
    dpi: int = 300,
    cmap_irradiance: str = "inferno",
    cmap_loss: str = "viridis_r",
    glb_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate summary.csv, summary.json, and report.pdf in `output_dir`.
    Returns a dict of the paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_publication_style()

    logger.info("Generating figures...")
    figures = generate_all_figures(
        report, results, output_dir, dpi=dpi, cmap_irradiance=cmap_irradiance, cmap_loss=cmap_loss, glb_path=glb_path,
    )

    logger.info("Writing summary.csv / summary.json...")
    csv_path = write_summary_csv(report, output_dir / "summary.csv")
    json_path = write_summary_json(report, output_dir / "summary.json")

    logger.info("Assembling PDF report...")
    pdf_path = output_dir / "report.pdf"
    observations = _generate_key_observations(report)

    with PdfPages(pdf_path) as pdf:
        run_names = ", ".join(f"{s.simulation_mode} ({s.source_path})" for s in report.single)
        _add_text_page(pdf, "PASA-Sim Analysis Report", [
            f"Run(s): {run_names}",
            "",
            "Executive summary:",
            *[f"  - {o}" for o in observations],
        ])

        for single, res in zip(report.single, results):
            _add_text_page(pdf, f"Run: {single.simulation_mode}", [
                f"Total energy yield: {single.total_energy_kwh:,.1f} kWh",
                f"Roof faces: overall stats -> {single.roof_stats}",
                f"Wall faces: overall stats -> {single.wall_stats}",
                f"Attenuation model: {res.metadata.get('attenuation_model_name')}",
                f"Latitude/Longitude: {res.metadata.get('latitude')}, {res.metadata.get('longitude')}",
                f"Simulated hours: {res.n_hours:,} | Faces: {res.n_faces:,}",
                f"Runtime: {res.metadata.get('runtime_seconds', 0):.1f}s | Ray backend: {res.metadata.get('ray_backend_used')}",
            ])
            _add_table_page(pdf, f"Top buildings -- {single.simulation_mode}", single.ranked_buildings["top"])
            _add_table_page(pdf, f"Bottom buildings -- {single.simulation_mode}", single.ranked_buildings["bottom"])

        if report.comparison is not None:
            c = report.comparison
            _add_text_page(pdf, "Clear-sky vs Pollution-aware comparison", [
                f"Total energy loss: {c.total_energy_loss_kwh:,.1f} kWh ({c.total_loss_pct:.1f}%)",
                f"Per-face loss %% stats: {c.loss_pct_stats}",
            ])
            _add_table_page(pdf, "Buildings most affected by pollution", c.building_comparison.head(20))

        for name, path in figures.items():
            if path.suffix == ".png":
                _add_figure_page(pdf, path, name.replace("_", " "))

    logger.info(f"Report complete: {pdf_path}, {csv_path}, {json_path}")
    return {"summary_csv": csv_path, "summary_json": json_path, "report_pdf": pdf_path, **figures}
