"""
analysis.py

Standalone analysis program AND importable library for PASA-Sim results.

    python -m pasa.analysis result.npz                     # single-run report
    python -m pasa.analysis clear.npz pollution.npz         # comparison report
    python analysis.py result.npz [result2.npz]             # also works stand-alone

Given one NPZ, computes the full descriptive-statistics suite (roof/wall/
building-level totals, distributions, orientation stats, shading, temporal
aggregates). Given two, additionally compares them -- detecting which is
the clear-sky run and which is the pollution run from each file's own
metadata (falling back to argument order if that's ambiguous) -- and
computes the pollution-driven energy loss.

This module computes NUMBERS only. Figures live in plotting.py; PDF/CSV/
JSON assembly lives in report.py. Running this file directly chains all
three, because that's what "a complete report" means to a user pointing
this at an NPZ from the command line.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io.npz_io import SimulationResult, load_simulation_result
from .utilities import get_logger, safe_divide, setup_logging

logger = get_logger()

DEFAULT_PERCENTILES: list[float] = [5, 25, 50, 75, 95]


# ---------------------------------------------------------------------------
# Small, reused primitives
# ---------------------------------------------------------------------------
def descriptive_stats(values: np.ndarray, percentiles: list[float] = DEFAULT_PERCENTILES) -> dict[str, Any]:
    """count/sum/mean/std/min/max/median/percentiles for a 1D array,
    ignoring NaN (which marks faces excluded from simulation). The single
    stats primitive every other function in this module calls, rather
    than recomputing mean/std/percentiles inline wherever they're needed.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        empty = {"count": 0, "sum": 0.0, "mean": float("nan"), "std": float("nan"),
                  "min": float("nan"), "max": float("nan"), "median": float("nan")}
        empty["percentiles"] = {f"p{p:g}": float("nan") for p in percentiles}
        return empty
    return {
        "count": int(values.size),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "percentiles": {f"p{p:g}": float(np.percentile(values, p)) for p in percentiles},
    }


def building_level_table(res: SimulationResult) -> pd.DataFrame:
    """One row per building: face counts, roof/wall/total area, total
    annual energy (irradiance x area, summed over faces), mean irradiance,
    and area-weighted specific yield (Wh/m^2/yr).
    """
    sim_mask = res.is_roof | res.is_wall
    df = pd.DataFrame({
        "building_id": res.building_ids()[sim_mask],
        "area_m2": res.area_m2[sim_mask],
        "is_roof": res.is_roof[sim_mask],
        "annual_irradiance_wh_m2": res.annual_irradiance_wh_m2[sim_mask],
    })
    df["annual_energy_wh"] = df["annual_irradiance_wh_m2"] * df["area_m2"]

    roof_area = df.loc[df.is_roof].groupby("building_id")["area_m2"].sum().rename("roof_area_m2")
    wall_area = df.loc[~df.is_roof].groupby("building_id")["area_m2"].sum().rename("wall_area_m2")
    totals = df.groupby("building_id").agg(
        n_faces=("area_m2", "size"),
        total_area_m2=("area_m2", "sum"),
        total_annual_energy_wh=("annual_energy_wh", "sum"),
        mean_annual_irradiance_wh_m2=("annual_irradiance_wh_m2", "mean"),
    )
    out = totals.join(roof_area, how="left").join(wall_area, how="left").fillna(0.0).reset_index()
    out["specific_yield_wh_m2"] = safe_divide(out["total_annual_energy_wh"].to_numpy(), out["total_area_m2"].to_numpy())
    return out.sort_values("total_annual_energy_wh", ascending=False).reset_index(drop=True)


def shading_table(res: SimulationResult) -> pd.DataFrame | None:
    """Per-building mean shaded fraction, restricted to daylight hours
    (elevation > 0) so night hours -- trivially "shadowed" for every face
    -- don't swamp the signal. Returns None if the NPZ didn't retain the
    hourly shadow mask (output.save_hourly_shadow_mask was False).
    """
    if res.hourly_shadow_mask is None:
        return None
    sim_mask = res.is_roof | res.is_wall
    daylight = res.solar_elevation_deg > 0
    if not daylight.any():
        return None
    shadow_daylight = res.hourly_shadow_mask[sim_mask][:, daylight]
    per_face_shaded_fraction = shadow_daylight.mean(axis=1)
    df = pd.DataFrame({"building_id": res.building_ids()[sim_mask], "shaded_fraction": per_face_shaded_fraction})
    return df.groupby("building_id")["shaded_fraction"].mean().reset_index().sort_values("shaded_fraction").reset_index(drop=True)


def temporal_aggregates(res: SimulationResult) -> dict[str, pd.Series]:
    """City-mean irradiance aggregated by month, season, hour-of-day, and
    calendar day -- the shared basis for monthly/seasonal/daily/hourly
    plots in plotting.py.
    """
    if res.hourly_irradiance_w_m2 is None:
        return {}
    sim_mask = res.is_roof | res.is_wall
    city_mean_w_m2 = np.nanmean(res.hourly_irradiance_w_m2[sim_mask], axis=0)  # (n_hours,)
    idx = res.times()
    s = pd.Series(city_mean_w_m2, index=idx)

    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    return {
        "monthly_mean_w_m2": s.groupby(s.index.month).mean(),
        "monthly_total_wh_m2": s.groupby(s.index.month).sum(),
        "seasonal_mean_w_m2": s.groupby(s.index.month.map(season_map)).mean(),
        "hourly_of_day_mean_w_m2": s.groupby(s.index.hour).mean(),
        "daily_mean_w_m2": s.groupby(s.index.date).mean(),
    }


def orientation_stats(res: SimulationResult) -> dict[str, Any]:
    """Tilt/azimuth distribution summaries for roofs and walls separately
    -- the numeric basis for the polar/rose-diagram plots."""
    out: dict[str, Any] = {}
    for label, mask in (("roof", res.is_roof), ("wall", res.is_wall)):
        out[f"{label}_tilt_deg"] = descriptive_stats(res.tilt_deg[mask])
        az = res.azimuth_deg[mask]
        az = az[~np.isnan(az)]
        out[f"{label}_azimuth_deg"] = descriptive_stats(az) if az.size else None
    return out


def ranked_faces(res: SimulationResult, top_n: int) -> dict[str, pd.DataFrame]:
    """Top/bottom N individual faces by annual irradiance -- finer-grained
    than the building-level ranking, useful for spotting specific
    problem/opportunity surfaces (e.g. a single heavily shaded roof
    corner) that a building-level average would hide.
    """
    sim_mask = res.is_roof | res.is_wall
    df = pd.DataFrame({
        "face_id": res.face_id[sim_mask],
        "building_id": res.building_ids()[sim_mask],
        "is_roof": res.is_roof[sim_mask],
        "tilt_deg": res.tilt_deg[sim_mask],
        "area_m2": res.area_m2[sim_mask],
        "annual_irradiance_wh_m2": res.annual_irradiance_wh_m2[sim_mask],
    }).sort_values("annual_irradiance_wh_m2", ascending=False)
    return {"top": df.head(top_n).reset_index(drop=True), "bottom": df.tail(top_n).iloc[::-1].reset_index(drop=True)}


# ---------------------------------------------------------------------------
# Top-level result containers
# ---------------------------------------------------------------------------
@dataclass
class SingleRunAnalysis:
    source_path: str
    simulation_mode: str
    overall_stats: dict[str, Any]
    roof_stats: dict[str, Any]
    wall_stats: dict[str, Any]
    orientation: dict[str, Any]
    building_table: pd.DataFrame
    shading_table: pd.DataFrame | None
    ranked_faces: dict[str, pd.DataFrame]
    ranked_buildings: dict[str, pd.DataFrame]
    temporal: dict[str, pd.Series]
    total_energy_kwh: float


@dataclass
class ComparisonAnalysis:
    clear: SingleRunAnalysis
    pollution: SingleRunAnalysis
    loss_wh_m2_stats: dict[str, Any]
    loss_pct_stats: dict[str, Any]
    total_energy_loss_kwh: float
    total_loss_pct: float
    per_face_loss_pct: np.ndarray
    building_comparison: pd.DataFrame


@dataclass
class AnalysisReport:
    single: list[SingleRunAnalysis] = field(default_factory=list)
    comparison: ComparisonAnalysis | None = None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyze_single(res: SimulationResult, top_n: int, percentiles: list[float]) -> SingleRunAnalysis:
    sim_mask = res.is_roof | res.is_wall
    building_table = building_level_table(res)
    total_energy_kwh = float(building_table["total_annual_energy_wh"].sum() / 1000.0)

    return SingleRunAnalysis(
        source_path=res.metadata.get("run_name", "unknown"),
        simulation_mode=res.metadata.get("simulation_mode", "unknown"),
        overall_stats=descriptive_stats(res.annual_irradiance_wh_m2[sim_mask], percentiles),
        roof_stats=descriptive_stats(res.annual_irradiance_wh_m2[res.is_roof], percentiles),
        wall_stats=descriptive_stats(res.annual_irradiance_wh_m2[res.is_wall], percentiles),
        orientation=orientation_stats(res),
        building_table=building_table,
        shading_table=shading_table(res),
        ranked_faces=ranked_faces(res, top_n),
        ranked_buildings={"top": building_table.head(top_n), "bottom": building_table.tail(top_n).iloc[::-1]},
        temporal=temporal_aggregates(res),
        total_energy_kwh=total_energy_kwh,
    )


def _pick_clear_and_pollution(res_a: SimulationResult, res_b: SimulationResult) -> tuple[SimulationResult, SimulationResult]:
    mode_a = res_a.metadata.get("simulation_mode")
    mode_b = res_b.metadata.get("simulation_mode")
    if {mode_a, mode_b} == {"clear", "pollution"}:
        return (res_a, res_b) if mode_a == "clear" else (res_b, res_a)
    logger.warning(
        f"Could not unambiguously identify clear-sky vs pollution runs from metadata "
        f"(modes found: {mode_a!r}, {mode_b!r}); assuming the first file given is the "
        "clear-sky baseline and the second is the pollution run."
    )
    return res_a, res_b


def _validate_comparison_alignment(a: SimulationResult, b: SimulationResult) -> None:
    """Reject positional comparisons of different time axes or city surfaces."""
    for name in ("time_unix_s", "face_id", "building_id_code", "is_roof", "is_wall"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise ValueError(f"Comparison requires matching {name} in both result files")
    for name in ("area_m2", "normal", "centroid"):
        x, y = getattr(a, name), getattr(b, name)
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True):
            raise ValueError(f"Comparison requires matching {name} in both result files")
    if a.building_id_lookup != b.building_id_lookup:
        raise ValueError("Comparison requires matching building_id_lookup in both result files")


def analyze_loaded(results: list[SimulationResult], top_n: int = 20, percentiles: list[float] = DEFAULT_PERCENTILES) -> AnalysisReport:
    """Same as analyze(), but takes already-loaded SimulationResult
    objects. Split out so callers that also need the raw results for
    plotting (report.py) don't have to read potentially-large NPZ files
    from disk twice.
    """
    if len(results) not in (1, 2):
        raise ValueError(f"analyze_loaded() expects 1 or 2 results, got {len(results)}")
    if len(results) == 2:
        _validate_comparison_alignment(results[0], results[1])

    singles = [analyze_single(r, top_n, percentiles) for r in results]
    if len(results) == 1:
        return AnalysisReport(single=singles)

    clear_res, pollution_res = _pick_clear_and_pollution(results[0], results[1])
    clear_single = singles[0] if results[0] is clear_res else singles[1]
    pollution_single = singles[0] if results[0] is pollution_res else singles[1]

    if clear_res.n_faces != pollution_res.n_faces:
        raise ValueError(
            f"Face count mismatch between the two NPZ files ({clear_res.n_faces} vs {pollution_res.n_faces}) "
            "-- are they runs of the same city model?"
        )

    sim_mask = (clear_res.is_roof | clear_res.is_wall) & (pollution_res.is_roof | pollution_res.is_wall)
    clear_annual = clear_res.annual_irradiance_wh_m2[sim_mask]
    pollution_annual = pollution_res.annual_irradiance_wh_m2[sim_mask]
    area = clear_res.area_m2[sim_mask]

    loss_wh_m2 = clear_annual - pollution_annual
    loss_pct = safe_divide(loss_wh_m2, clear_annual) * 100.0

    total_clear_kwh = float(np.nansum(clear_annual * area) / 1000.0)
    total_pollution_kwh = float(np.nansum(pollution_annual * area) / 1000.0)
    total_loss_kwh = total_clear_kwh - total_pollution_kwh
    total_loss_pct = 100.0 * total_loss_kwh / total_clear_kwh if total_clear_kwh > 0 else float("nan")

    clear_bt = clear_single.building_table.set_index("building_id")
    pollution_bt = pollution_single.building_table.set_index("building_id")
    building_comparison = clear_bt[["total_annual_energy_wh", "total_area_m2"]].rename(
        columns={"total_annual_energy_wh": "clear_energy_wh"}
    ).join(
        pollution_bt[["total_annual_energy_wh"]].rename(columns={"total_annual_energy_wh": "pollution_energy_wh"}),
        how="inner",
    )
    building_comparison["loss_wh"] = building_comparison["clear_energy_wh"] - building_comparison["pollution_energy_wh"]
    building_comparison["loss_pct"] = safe_divide(
        building_comparison["loss_wh"].to_numpy(), building_comparison["clear_energy_wh"].to_numpy()
    ) * 100.0
    building_comparison = building_comparison.reset_index().sort_values("loss_pct", ascending=False)

    comparison = ComparisonAnalysis(
        clear=clear_single, pollution=pollution_single,
        loss_wh_m2_stats=descriptive_stats(loss_wh_m2, percentiles),
        loss_pct_stats=descriptive_stats(loss_pct, percentiles),
        total_energy_loss_kwh=total_loss_kwh,
        total_loss_pct=total_loss_pct,
        per_face_loss_pct=loss_pct,
        building_comparison=building_comparison,
    )
    return AnalysisReport(single=singles, comparison=comparison)


def analyze(paths: list[str | Path], top_n: int = 20, percentiles: list[float] = DEFAULT_PERCENTILES) -> AnalysisReport:
    if len(paths) not in (1, 2):
        raise ValueError(f"analyze() expects 1 or 2 NPZ paths, got {len(paths)}")
    results = [load_simulation_result(p) for p in paths]
    return analyze_loaded(results, top_n=top_n, percentiles=percentiles)


# ---------------------------------------------------------------------------
# CLI entry point (also importable/runnable as a genuinely standalone script)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze one PASA-Sim result, or compare two.")
    parser.add_argument("npz_paths", nargs="+", help="One NPZ for a single-run report, or two (clear + pollution) to compare")
    parser.add_argument("--output-dir", default="analysis_output", help="Where to write summary.csv/json, figures, and the PDF report")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--glb-path", default=None, help="Original city GLB, for solid-mesh (not point-cloud) 3D figures")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution (publication default: 300)")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    if len(args.npz_paths) not in (1, 2):
        parser.error("Provide exactly 1 (single-run report) or 2 (comparison report) NPZ paths")

    results = [load_simulation_result(p) for p in args.npz_paths]
    report = analyze_loaded(results, top_n=args.top_n)

    # Imported here, not at module level, so `python analysis.py` doesn't pay the
    # (heavier) matplotlib/plotly import cost for users who only want the numbers
    # via `from pasa.analysis import analyze` in their own notebook/script.
    from .report import generate_report

    generate_report(report, results, Path(args.output_dir), dpi=args.dpi, glb_path=args.glb_path)


if __name__ == "__main__":
    main(sys.argv[1:])
