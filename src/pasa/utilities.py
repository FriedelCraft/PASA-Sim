"""
utilities.py

Small, dependency-light helpers shared across the framework: logging setup,
a timing context manager, a stable config hash (used to validate checkpoint
resumability), the local ENU -> lat/lon approximation, and a couple of
numerically-safe math helpers. Nothing in this file knows about geometry,
solar physics, or simulation state -- it exists specifically so that kind
of logic is written once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EARTH_RADIUS_M

_LOGGER_NAME = "pasa"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure and return the framework-wide 'pasa' logger.

    Safe to call multiple times (e.g. once from the CLI, once from a
    notebook) -- existing handlers are cleared first so log lines are never
    duplicated.

    Parameters
    ----------
    level:
        One of "DEBUG", "INFO", "WARNING", "ERROR".
    log_file:
        If given, logs are additionally written to this file (parent
        directories are created as needed). Console logging always happens.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the framework logger, configuring a default console-only
    logger the first time this is called if setup_logging() was never
    invoked (e.g. when a module is imported directly for testing).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        return setup_logging()
    return logger


class Timer:
    """Context manager that logs the wall-clock duration of a code block.

    Example
    -------
    >>> with Timer("ray tracing chunk"):
    ...     do_expensive_thing()
    """

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger or get_logger()
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_seconds = time.perf_counter() - self._t0
        if exc_type is None:
            self.logger.info(f"{self.label}: {self.elapsed_seconds:.2f}s")
        else:
            self.logger.warning(f"{self.label}: FAILED after {self.elapsed_seconds:.2f}s ({exc_type.__name__}: {exc_val})")


def hash_config(config_dict: dict[str, Any]) -> str:
    """Stable, order-independent hash of a (nested) config dict.

    Used by checkpoint.CheckpointManager to detect whether a checkpoint on
    disk was produced by a materially different configuration (different
    city model, different simulation mode, different time range, ...) than
    the one currently being run, so a mismatched checkpoint is never
    silently resumed into.
    """
    blob = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def git_commit_hash(cwd: str | Path | None = None) -> str:
    """Best-effort current git commit hash, for reproducibility metadata
    embedded in output NPZ files. Returns "unknown" if the working
    directory isn't a git repo or git isn't available -- this must never
    raise, since it's only ever used for an informational metadata field.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def wrap_angle_deg(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) in degrees into [0, 360)."""
    return np.mod(angle, 360.0)


def enu_to_latlon(
    east_m: np.ndarray | float,
    north_m: np.ndarray | float,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local flat-Earth approximation mapping ENU offsets (meters, relative
    to a reference point) to geographic latitude/longitude (degrees).

    This is a small-angle / equirectangular approximation: it treats the
    Earth as a sphere of radius EARTH_RADIUS_M and assumes the AOI is small
    enough that curvature and ellipsoidal effects are negligible. Valid to
    well under 0.5% error for AOI extents up to a few tens of km, which is
    the intended use case here (matching city-model face centroids to a
    coarser-resolution AOD raster/point dataset). Do not reuse this for
    large-extent domains -- use a proper projection library instead.

    Parameters
    ----------
    east_m, north_m:
        East and North offsets in meters from the (lat0_deg, lon0_deg)
        reference point. Scalars or same-shaped arrays.
    lat0_deg, lon0_deg:
        Geographic coordinates of the ENU frame's origin.

    Returns
    -------
    (lat_deg, lon_deg) arrays/scalars matching the shape of the inputs.
    """
    east_m = np.asarray(east_m, dtype=np.float64)
    north_m = np.asarray(north_m, dtype=np.float64)
    lat_deg = lat0_deg + np.degrees(north_m / EARTH_RADIUS_M)
    lon_deg = lon0_deg + np.degrees(east_m / (EARTH_RADIUS_M * np.cos(np.radians(lat0_deg))))
    return lat_deg, lon_deg


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Elementwise division that replaces division-by-zero results with
    `fill` instead of raising a RuntimeWarning / producing inf or NaN, e.g.
    for per-building means where a building might (in principle) have zero
    contributing faces after exclusion filters.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / np.where(denominator == 0, 1, denominator), fill)
    return result


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split ``range(total)`` into contiguous [start, end) chunks of at
    most ``chunk_size``. The single loop-chunking primitive used by
    simulation.py for the hour-chunk loop, checkpoint.py for resumability,
    and (indirectly) the multiprocessing task split -- written once here so
    all three agree on exactly the same chunk boundaries.
    """
    if total <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]
