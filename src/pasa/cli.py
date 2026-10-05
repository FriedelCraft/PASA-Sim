"""
cli.py

Unified command-line entry point (replaces the old main.py):

    pasa simulate --config config.yaml
    pasa analyze outputs/clear.npz outputs/pollution.npz --output-dir report/

Installed as a console script (`pasa`) by pyproject.toml; also runnable as
`python -m pasa.cli` without installing the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, PASAConfig
from .utilities import get_logger, setup_logging

logger = get_logger()


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .simulation import run_simulation

    try:
        config = PASAConfig.from_yaml(args.config, simulation_mode=args.mode)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.output_dir:
        config.run.output_dir = args.output_dir
    if args.workers is not None:
        config.performance.n_workers = args.workers
    config.validate()

    try:
        result_path = run_simulation(config)
    except Exception as exc:
        logger.error(f"Simulation failed: {exc}")
        raise
    print(f"Wrote {result_path}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze_loaded
    from .io.npz_io import load_simulation_result
    from .report import generate_report

    if len(args.npz_paths) not in (1, 2):
        print("Provide exactly 1 (single-run report) or 2 (comparison report) NPZ paths", file=sys.stderr)
        return 2

    top_n, percentiles = args.top_n, None
    dpi, cmap_irradiance, cmap_loss = args.dpi, "inferno", "viridis_r"
    if args.config:
        try:
            config = PASAConfig.from_yaml(args.config)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        # Explicit CLI flags win over the config file; the config file wins over
        # this function's own hardcoded fallback defaults above.
        top_n = args.top_n if args.top_n is not None else config.analysis.top_n_buildings
        percentiles = config.analysis.percentiles
        dpi = args.dpi if args.dpi is not None else config.plotting.dpi
        cmap_irradiance = config.plotting.colormap_irradiance
        cmap_loss = config.plotting.colormap_loss
    top_n = top_n if top_n is not None else 20
    dpi = dpi if dpi is not None else 300
    percentiles = percentiles if percentiles is not None else [5, 25, 50, 75, 95]

    results = [load_simulation_result(p) for p in args.npz_paths]
    report = analyze_loaded(results, top_n=top_n, percentiles=percentiles)
    paths = generate_report(
        report, results, Path(args.output_dir), dpi=dpi,
        cmap_irradiance=cmap_irradiance, cmap_loss=cmap_loss, glb_path=args.glb_path,
    )
    print(f"Report written to {args.output_dir}:")
    for name in ("summary_csv", "summary_json", "report_pdf"):
        print(f"  {name}: {paths[name]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pasa", description="PASA-Sim: Pollution-Aware Solar Assessment simulator")
    parser.add_argument("--version", action="version", version=f"PASA-Sim {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("simulate", help="Run a clear-sky or pollution-aware simulation")
    sim_parser.add_argument("--config", required=True, help="Path to a YAML config file (see config.yaml)")
    sim_parser.add_argument("--mode", choices=("clear", "pollution"), default=None, help="Override run.simulation_mode from the config")
    sim_parser.add_argument("--output-dir", default=None, help="Override run.output_dir from the config")
    sim_parser.add_argument("--workers", type=int, default=None, help="Override performance.n_workers from the config")
    sim_parser.set_defaults(func=_cmd_simulate)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one result, or compare two (clear vs pollution)")
    analyze_parser.add_argument("npz_paths", nargs="+")
    analyze_parser.add_argument("--config", default=None, help="Optional YAML config to source analysis.*/plotting.* defaults from (see config.yaml)")
    analyze_parser.add_argument("--output-dir", default="analysis_output")
    analyze_parser.add_argument("--top-n", type=int, default=None, help="Default: analysis.top_n_buildings from --config, else 20")
    analyze_parser.add_argument("--dpi", type=int, default=None, help="Default: plotting.dpi from --config, else 300")
    analyze_parser.add_argument("--glb-path", default=None, help="Original city GLB, for solid-mesh 3D figures")
    analyze_parser.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
