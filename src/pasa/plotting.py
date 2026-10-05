"""
plotting.py

Figure generation for PASA-Sim results. Every function takes already-
computed data (numpy arrays / pandas objects from analysis.py or the raw
SimulationResult) and writes exactly one figure file, at
config.plotting.dpi, using a single shared style (`setup_publication_style`)
so every figure -- whether it's an irradiance map or a rose diagram --
shares fonts, DPI, and colorbar conventions without each function
re-implementing that formatting.

Rather than one function per item in a long "produce every plot type"
wishlist, a handful of general, parameterized primitives cover the same
ground: `plot_distribution` is histogram + KDE + PDF in one (those are the
same underlying computation applied to a 1D array), `plot_box_comparison`
covers both "roof vs wall" and "clear vs pollution", `plot_ranking_bar`
covers best/worst/top-20/bottom-20 for both faces and buildings, etc.
`generate_all_figures` is the single orchestration entry point report.py
calls; it decides which figures apply (e.g. a loss map only makes sense
for a comparison) and interleaves calls to the primitives below.

3D mesh rendering note: the output NPZ deliberately does not store
vertex/triangle-index arrays (redundant with the GLB file that already
exists on disk, and expensive at 100k+ faces). The solid-mesh figures
below reconstruct real triangle geometry by re-reading the original GLB
via io.mesh_io/geometry (using the exact same deterministic face-build
order, so face i in the NPZ is guaranteed to be face i of the reloaded
mesh) when `glb_path` is supplied; without it, they fall back to a
centroid point-cloud rendering, which needs no extra file but reads as a
scatter of points rather than a solid surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats as scipy_stats

from .io.npz_io import SimulationResult
from .utilities import get_logger

logger = get_logger()

_STYLE_APPLIED = False


def setup_publication_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    _STYLE_APPLIED = True


def _save(fig: plt.Figure, output_path: Path, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _load_mesh_geometry(glb_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Reload (vertices, faces) from the original GLB, in the same
    deterministic order geometry.build_city_geometry produces -- so face i
    here is guaranteed to be face i in any SimulationResult built from the
    same file.
    """
    from .io.mesh_io import load_raw_city_mesh

    raw = load_raw_city_mesh(glb_path)
    return raw.vertices, raw.faces


# ---------------------------------------------------------------------------
# 3D city mesh -- the workhorse behind the irradiance map, roof/wall heat
# map, pollution loss map, and difference map: same function, different
# `values` array and colormap.
# ---------------------------------------------------------------------------
def plot_city_mesh_3d(
    face_values: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    cmap: str = "inferno",
    centroids: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    glb_path: str | Path | None = None,
    diverging: bool = False,
) -> Path:
    """Render the city colored by `face_values` (one value per face,
    NaN for faces to omit). Uses a true solid Poly3DCollection mesh if
    (vertices, faces) are given or can be loaded via `glb_path`; otherwise
    falls back to a centroid point cloud (`centroids` required in that
    case).
    """
    setup_publication_style()

    if vertices is None and glb_path is not None:
        vertices, faces = _load_mesh_geometry(glb_path)

    valid = ~np.isnan(face_values)
    norm = TwoSlopeNorm(vcenter=0, vmin=min(0, np.nanmin(face_values)), vmax=max(1e-6, np.nanmax(face_values))) \
        if diverging else Normalize(vmin=np.nanmin(face_values[valid]), vmax=np.nanmax(face_values[valid]))
    colormap = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if vertices is not None and faces is not None:
        tri_verts = vertices[faces][valid]  # (n_valid, 3, 3)
        facecolors = colormap(norm(face_values[valid]))
        collection = Poly3DCollection(tri_verts, facecolors=facecolors, edgecolors="none", linewidths=0)
        ax.add_collection3d(collection)
        mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    else:
        if centroids is None:
            raise ValueError("plot_city_mesh_3d needs either (vertices, faces)/glb_path, or centroids")
        ax.scatter(
            centroids[valid, 0], centroids[valid, 1], centroids[valid, 2],
            c=face_values[valid], cmap=cmap, norm=norm, s=8, depthshade=True,
        )
        mins, maxs = centroids.min(axis=0), centroids.max(axis=0)

    span = (maxs - mins).max() / 2
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(max(0, mid[2] - span), mid[2] + span)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-60)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=colorbar_label)

    return _save(fig, output_path, dpi)


def plot_city_mesh_interactive(
    face_values: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str = "inferno",
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    glb_path: str | Path | None = None,
) -> Path | None:
    """Plotly Mesh3d version of plot_city_mesh_3d, saved as a self-
    contained HTML file the user can rotate/zoom/hover in a browser --
    the "interactive visualization" requirement. Returns None (and logs a
    warning) if no mesh geometry is available, since a meaningful
    interactive city view needs the real triangles, not a point cloud.
    """
    import plotly.graph_objects as go

    if vertices is None and glb_path is not None:
        vertices, faces = _load_mesh_geometry(glb_path)
    if vertices is None or faces is None:
        logger.warning(f"Skipping interactive mesh '{title}': no GLB path available to reload mesh geometry")
        return None

    valid = ~np.isnan(face_values)
    norm = Normalize(vmin=np.nanmin(face_values[valid]), vmax=np.nanmax(face_values[valid]))
    colormap = plt.get_cmap(cmap)
    facecolor = np.full(len(faces), "rgba(0,0,0,0)", dtype=object)
    rgba = (colormap(norm(face_values[valid])) * 255).astype(int)
    facecolor[valid] = [f"rgb({r},{g},{b})" for r, g, b, _ in rgba]

    mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor=list(facecolor), flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Up (m)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


# ---------------------------------------------------------------------------
# General-purpose statistical primitives (each reused across several of the
# requested plot types -- see module docstring).
# ---------------------------------------------------------------------------
def plot_distribution(
    values: np.ndarray, title: str, xlabel: str, output_path: Path, dpi: int, bins: int = 40, color: str = "#d1495b",
) -> Path:
    """Histogram + KDE overlay: covers "histogram", "distribution plot",
    "PDF", and "kernel density" in one figure, since for a continuous 1D
    sample these are the same information shown two complementary ways.
    """
    setup_publication_style()
    values = values[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, density=True, color=color, alpha=0.55, edgecolor="white", linewidth=0.4, label="Histogram")
    if values.size > 1 and np.std(values) > 0:
        kde = scipy_stats.gaussian_kde(values)
        xs = np.linspace(values.min(), values.max(), 300)
        ax.plot(xs, kde(xs), color="#2b2d42", linewidth=1.8, label="KDE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save(fig, output_path, dpi)


def plot_cdf(values: np.ndarray, title: str, xlabel: str, output_path: Path, dpi: int, color: str = "#2b2d42") -> Path:
    setup_publication_style()
    values = np.sort(values[~np.isnan(values)])
    y = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, y, color=color, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(title)
    return _save(fig, output_path, dpi)


def plot_box_comparison(
    groups: dict[str, np.ndarray], title: str, ylabel: str, output_path: Path, dpi: int,
) -> Path:
    """Covers "roof vs wall", "clear vs pollution", or any other small set
    of named distributions to compare side by side.
    """
    setup_publication_style()
    labels = list(groups.keys())
    data = [g[~np.isnan(g)] for g in groups.values()]
    fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(labels)), 4.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5)
    palette = ["#d1495b", "#2b2d42", "#118ab2", "#f4a261", "#06a77d"]
    for patch, color in zip(bp["boxes"], palette * (len(labels) // len(palette) + 1)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save(fig, output_path, dpi)


def plot_scatter(
    x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, output_path: Path, dpi: int,
    color_by: np.ndarray | None = None, color_label: str = "", cmap: str = "viridis",
) -> Path:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if color_by is not None:
        sc = ax.scatter(x, y, c=color_by, cmap=cmap, s=10, alpha=0.75, edgecolors="none")
        fig.colorbar(sc, ax=ax, label=color_label)
    else:
        ax.scatter(x, y, s=10, alpha=0.6, color="#118ab2", edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save(fig, output_path, dpi)


def plot_correlation_heatmap(df: pd.DataFrame, title: str, output_path: Path, dpi: int) -> Path:
    setup_publication_style()
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(corr), 1 + 0.6 * len(corr)))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(title)
    return _save(fig, output_path, dpi)


def plot_bar_series(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path, dpi: int, color: str = "#f4a261") -> Path:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(series.index.astype(str), series.to_numpy(), color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, output_path, dpi)


def plot_line_series(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path, dpi: int, color: str = "#2b2d42") -> Path:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    is_numeric_index = np.issubdtype(np.asarray(series.index).dtype, np.number)
    x_vals = series.index if is_numeric_index else series.index.astype(str)
    ax.plot(x_vals, series.to_numpy(), color=color, linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if series.index.size > 15:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, output_path, dpi)


def plot_sunrise_sunset(res: SimulationResult, output_path: Path, dpi: int) -> Path | None:
    """Daily sunrise/sunset time band across the simulated period,
    derived from where solar elevation crosses zero each day.
    """
    setup_publication_style()
    idx = res.times()
    elevation = pd.Series(res.solar_elevation_deg, index=idx)
    daily = elevation.groupby(idx.date)

    rows = []
    for day, series in daily:
        above = series[series > 0]
        if above.empty:
            continue
        rows.append({"day": day, "sunrise_hour": above.index.min().hour + above.index.min().minute / 60,
                      "sunset_hour": above.index.max().hour + above.index.max().minute / 60})
    if not rows:
        logger.warning("No daylight hours found in the simulated period; skipping sunrise/sunset plot")
        return None
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(df["day"].astype(str), df["sunrise_hour"], df["sunset_hour"], color="#f4a261", alpha=0.5, label="Daylight")
    ax.plot(df["day"].astype(str), df["sunrise_hour"], color="#e76f51", linewidth=1.2, label="Sunrise")
    ax.plot(df["day"].astype(str), df["sunset_hour"], color="#2b2d42", linewidth=1.2, label="Sunset")
    ax.set_ylabel("Hour of day (local time)")
    ax.set_title("Sunrise / sunset profile")
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, output_path, dpi)


def plot_polar_rose(
    azimuth_deg: np.ndarray, weights: np.ndarray, title: str, output_path: Path, dpi: int,
    n_bins: int = 16, color: str = "#118ab2",
) -> Path:
    """Rose diagram of face orientation (roof or wall azimuths), weighted
    by e.g. face area or annual irradiance -- compass convention (0=N,
    clockwise), matching pvlib's azimuth convention used everywhere else
    in this framework.
    """
    setup_publication_style()
    mask = ~np.isnan(azimuth_deg)
    az = np.radians(azimuth_deg[mask])
    w = weights[mask]
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    totals, _ = np.histogram(az, bins=bin_edges, weights=w)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(centers, totals, width=2 * np.pi / n_bins * 0.9, color=color, alpha=0.75, edgecolor="white")
    ax.set_title(title, pad=20)
    return _save(fig, output_path, dpi)


def plot_ranking_bar(
    labels: list[str], values: np.ndarray, title: str, xlabel: str, output_path: Path, dpi: int, color: str = "#06a77d",
) -> Path:
    """Horizontal bar ranking -- covers best/worst/top-N/bottom-N for
    both buildings and individual faces.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=color, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Orchestration: the one entry point report.py calls. Interleaves calls to
# the primitives above, deciding which figures apply to a single-run vs a
# comparison report, and returns {figure_name: path} for report.py to embed.
# ---------------------------------------------------------------------------
def generate_all_figures(
    report: "AnalysisReport",  # noqa: F821 (avoid a circular import; analysis.py imports plotting lazily too)
    results: list[SimulationResult],
    output_dir: Path,
    dpi: int = 300,
    cmap_irradiance: str = "inferno",
    cmap_loss: str = "viridis_r",
    glb_path: str | Path | None = None,
) -> dict[str, Path]:
    figures_dir = Path(output_dir) / "figures"
    figures: dict[str, Path] = {}

    def _add(name: str, path: Path | None) -> None:
        if path is not None:
            figures[name] = path

    for single, res in zip(report.single, results):
        prefix = single.simulation_mode or "run"
        sim_mask = res.is_roof | res.is_wall

        _add(f"{prefix}_annual_irradiance_map", plot_city_mesh_3d(
            res.annual_irradiance_wh_m2, f"Annual irradiance -- {prefix}", "Wh/m^2",
            figures_dir / f"{prefix}_annual_irradiance_map.png", dpi, cmap=cmap_irradiance,
            centroids=res.centroid, glb_path=glb_path,
        ))
        _add(f"{prefix}_annual_irradiance_map_interactive", plot_city_mesh_interactive(
            res.annual_irradiance_wh_m2, f"Annual irradiance -- {prefix}",
            figures_dir / f"{prefix}_annual_irradiance_map.html", cmap=cmap_irradiance, glb_path=glb_path,
        ))
        _add(f"{prefix}_roof_wall_distribution", plot_box_comparison(
            {"Roof": res.annual_irradiance_wh_m2[res.is_roof], "Wall": res.annual_irradiance_wh_m2[res.is_wall]},
            f"Roof vs wall annual irradiance -- {prefix}", "Wh/m^2",
            figures_dir / f"{prefix}_roof_wall_box.png", dpi,
        ))
        _add(f"{prefix}_irradiance_distribution", plot_distribution(
            res.annual_irradiance_wh_m2[sim_mask], f"Annual irradiance distribution -- {prefix}", "Wh/m^2",
            figures_dir / f"{prefix}_irradiance_hist.png", dpi,
        ))
        _add(f"{prefix}_irradiance_cdf", plot_cdf(
            res.annual_irradiance_wh_m2[sim_mask], f"Annual irradiance CDF -- {prefix}", "Wh/m^2",
            figures_dir / f"{prefix}_irradiance_cdf.png", dpi,
        ))
        _add(f"{prefix}_roof_azimuth_rose", plot_polar_rose(
            res.azimuth_deg[res.is_wall], res.area_m2[res.is_wall], f"Wall orientation (area-weighted) -- {prefix}",
            figures_dir / f"{prefix}_wall_azimuth_rose.png", dpi,
        ))
        _add(f"{prefix}_building_ranking_top", plot_ranking_bar(
            single.ranked_buildings["top"]["building_id"].tolist(),
            single.ranked_buildings["top"]["total_annual_energy_wh"].to_numpy() / 1000.0,
            f"Top {len(single.ranked_buildings['top'])} buildings by annual energy -- {prefix}", "kWh",
            figures_dir / f"{prefix}_building_ranking_top.png", dpi, color="#06a77d",
        ))
        _add(f"{prefix}_building_ranking_bottom", plot_ranking_bar(
            single.ranked_buildings["bottom"]["building_id"].tolist(),
            single.ranked_buildings["bottom"]["total_annual_energy_wh"].to_numpy() / 1000.0,
            f"Bottom {len(single.ranked_buildings['bottom'])} buildings by annual energy -- {prefix}", "kWh",
            figures_dir / f"{prefix}_building_ranking_bottom.png", dpi, color="#d1495b",
        ))

        if single.shading_table is not None and len(single.shading_table) > 0:
            st = single.shading_table
            _add(f"{prefix}_shading_ranking", plot_ranking_bar(
                st["building_id"].tolist(), st["shaded_fraction"].to_numpy() * 100,
                f"Daylight-hours shaded fraction by building -- {prefix}", "% of daylight hours shadowed",
                figures_dir / f"{prefix}_shading_ranking.png", dpi, color="#e76f51",
            ))

        if single.temporal:
            _add(f"{prefix}_monthly_totals", plot_bar_series(
                single.temporal["monthly_total_wh_m2"], f"Monthly total irradiance -- {prefix}", "Month", "Wh/m^2",
                figures_dir / f"{prefix}_monthly_totals.png", dpi,
            ))
            _add(f"{prefix}_seasonal_totals", plot_bar_series(
                single.temporal["seasonal_mean_w_m2"], f"Seasonal mean irradiance -- {prefix}", "Season", "W/m^2",
                figures_dir / f"{prefix}_seasonal_totals.png", dpi, color="#118ab2",
            ))
            _add(f"{prefix}_hourly_profile", plot_line_series(
                single.temporal["hourly_of_day_mean_w_m2"], f"Mean diurnal profile -- {prefix}", "Hour of day", "W/m^2",
                figures_dir / f"{prefix}_hourly_profile.png", dpi,
            ))
            _add(f"{prefix}_daily_profile", plot_line_series(
                single.temporal["daily_mean_w_m2"], f"Daily mean irradiance -- {prefix}", "Date", "W/m^2",
                figures_dir / f"{prefix}_daily_profile.png", dpi,
            ))

        _add(f"{prefix}_sunrise_sunset", plot_sunrise_sunset(res, figures_dir / f"{prefix}_sunrise_sunset.png", dpi))

        corr_df = pd.DataFrame({
            "tilt_deg": res.tilt_deg[sim_mask], "area_m2": res.area_m2[sim_mask],
            "annual_irradiance_wh_m2": res.annual_irradiance_wh_m2[sim_mask],
        })
        _add(f"{prefix}_correlation_heatmap", plot_correlation_heatmap(
            corr_df, f"Correlation matrix -- {prefix}", figures_dir / f"{prefix}_correlation.png", dpi,
        ))
        _add(f"{prefix}_tilt_vs_irradiance", plot_scatter(
            res.tilt_deg[sim_mask], res.annual_irradiance_wh_m2[sim_mask],
            f"Tilt vs annual irradiance -- {prefix}", "Tilt (deg)", "Wh/m^2",
            figures_dir / f"{prefix}_tilt_vs_irradiance.png", dpi,
            color_by=res.azimuth_deg[sim_mask], color_label="Azimuth (deg)",
        ))

    if report.comparison is not None:
        comp = report.comparison
        clear_res, pollution_res = (results[0], results[1]) if report.single[0] is comp.clear else (results[1], results[0])
        sim_mask = (clear_res.is_roof | clear_res.is_wall) & (pollution_res.is_roof | pollution_res.is_wall)

        loss_wh_m2_full = np.full(clear_res.n_faces, np.nan, dtype=np.float32)
        loss_wh_m2_full[sim_mask] = clear_res.annual_irradiance_wh_m2[sim_mask] - pollution_res.annual_irradiance_wh_m2[sim_mask]
        loss_pct_full = np.full(clear_res.n_faces, np.nan, dtype=np.float32)
        loss_pct_full[sim_mask] = comp.per_face_loss_pct

        _add("pollution_loss_map_wh_m2", plot_city_mesh_3d(
            loss_wh_m2_full, "Pollution-driven energy loss", "Wh/m^2 lost",
            figures_dir / "pollution_loss_map.png", dpi, cmap=cmap_loss,
            centroids=clear_res.centroid, glb_path=glb_path,
        ))
        _add("pollution_loss_map_pct", plot_city_mesh_3d(
            loss_pct_full, "Pollution-driven energy loss (%)", "% lost",
            figures_dir / "pollution_loss_pct_map.png", dpi, cmap=cmap_loss,
            centroids=clear_res.centroid, glb_path=glb_path,
        ))
        _add("clear_vs_pollution_box", plot_box_comparison(
            {"Clear sky": clear_res.annual_irradiance_wh_m2[sim_mask], "Pollution-aware": pollution_res.annual_irradiance_wh_m2[sim_mask]},
            "Clear-sky vs pollution-aware annual irradiance", "Wh/m^2",
            figures_dir / "clear_vs_pollution_box.png", dpi,
        ))
        _add("loss_pct_distribution", plot_distribution(
            comp.per_face_loss_pct, "Distribution of pollution-driven energy loss", "% lost",
            figures_dir / "loss_pct_distribution.png", dpi,
        ))
        bc_sorted = comp.building_comparison.sort_values("loss_pct", ascending=False)
        _add("building_loss_ranking", plot_ranking_bar(
            bc_sorted["building_id"].head(20).tolist(), bc_sorted["loss_pct"].head(20).to_numpy(),
            "Buildings most affected by pollution (% energy lost)", "% lost",
            figures_dir / "building_loss_ranking.png", dpi, color="#d1495b",
        ))
        _add("aod_vs_loss_scatter", plot_scatter(
            np.nanmean(pollution_res.aod_group_value, axis=1)[pollution_res.aod_group_id[sim_mask]]
                if pollution_res.aod_group_value is not None else np.zeros(sim_mask.sum()),
            comp.per_face_loss_pct,
            "Mean AOD vs energy loss per face", "Mean AOD (550 nm)", "% energy lost",
            figures_dir / "aod_vs_loss_scatter.png", dpi,
        ))

    logger.info(f"Generated {len(figures)} figure(s) in {figures_dir}")
    return figures
