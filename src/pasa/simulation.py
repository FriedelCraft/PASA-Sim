"""
simulation.py

Orchestrates a full PASA-Sim run: build geometry + solar + attenuation
model, loop over the year in hour-chunks (vectorized within a chunk,
optionally parallel across chunks), checkpoint periodically, assemble the
result, and write the output NPZ. This file contains no physics of its
own -- every physics/geometry computation is delegated to geometry.py,
solar.py, shadow.py, irradiance.py, and models/ -- its only job is calling
them in the right order and managing the loop.

This is the ONE place `config.run.simulation_mode` is read to decide
behavior (via `build_attenuation_model`'s registry lookup): everything
downstream of that single factory call is mode-agnostic.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import __version__
from . import models
from .checkpoint import CheckpointManager
from .config import PASAConfig
from .geometry import CityGeometry, build_city_geometry
from .io.aod_loaders import load_aod
from .io.mesh_io import load_raw_city_mesh
from .io.npz_io import SimulationResult, save_simulation_result
from .irradiance import compute_chunk_irradiance
from .models.base import BaseAttenuationModel
from .shadow import RayBackend, build_ray_backend
from .solar import SolarTimeSeries, build_time_index, compute_solar_time_series
from .utilities import chunk_ranges, get_logger, git_commit_hash, hash_config, setup_logging

logger = get_logger()

import trimesh  # noqa: E402  (after first-party imports is fine; trimesh is only needed for the raw-mesh wrapper below)


@dataclass
class SimulationContext:
    geometry: CityGeometry
    solar_ts: SolarTimeSeries
    attenuation_model: BaseAttenuationModel
    ray_backend: RayBackend
    sim_face_mask: np.ndarray
    face_to_group_id_full: np.ndarray
    ray_offset_m: float


def build_attenuation_model(config: PASAConfig, geometry: CityGeometry, times: pd.DatetimeIndex) -> BaseAttenuationModel:
    """The factory: construct whichever BaseAttenuationModel subclass
    `config.pollution.attenuation_model` names. This is the only function
    in the whole codebase that branches on the model registry key -- every
    other module just calls the resulting object's interface methods.
    """
    model_cls = models.get_model_class(config.pollution.attenuation_model)
    if config.pollution.attenuation_model == "clear":
        return model_cls()
    aod_dataset = load_aod(config.pollution.aod)
    return model_cls(
        aod_dataset, geometry, config.location, config.pollution.aod, times, config.solar.airmass_model,
    )


def build_simulation_context(config: PASAConfig) -> SimulationContext:
    raw = load_raw_city_mesh(config.city_model.glb_path)
    geometry = build_city_geometry(raw, config.city_model)

    times = build_time_index(config.time, config.location.timezone)
    solar_ts = compute_solar_time_series(times, config.location, config.solar)

    attenuation_model = build_attenuation_model(config, geometry, times)

    raw_mesh = trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces, process=False)
    ray_backend = build_ray_backend(geometry, raw_mesh, config.performance.use_embree_if_available)

    ray_offset_m = config.city_model.ray_offset_scale * geometry.bbox_diagonal_m

    return SimulationContext(
        geometry=geometry,
        solar_ts=solar_ts,
        attenuation_model=attenuation_model,
        ray_backend=ray_backend,
        sim_face_mask=geometry.simulated_mask(),
        face_to_group_id_full=attenuation_model.face_to_group_id(geometry.n_faces),
        ray_offset_m=ray_offset_m,
    )


# ---------------------------------------------------------------------------
# Multiprocessing worker: each worker process builds its OWN SimulationContext
# once (including its own Embree intersector and, for pollution runs, its own
# copy of the interpolated AOD series -- ray intersectors and pvlib/xarray
# handles are not picklable, so they cannot be built once and shared; this
# is a deliberate, documented one-time-per-worker cost, negligible next to
# the ray-tracing work each worker then does across many chunks).
# ---------------------------------------------------------------------------
_worker_ctx: SimulationContext | None = None


def _worker_init(config: PASAConfig) -> None:
    global _worker_ctx
    setup_logging(config.run.log_level)
    _worker_ctx = build_simulation_context(config)


def _worker_compute_chunk(chunk: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    assert _worker_ctx is not None, "worker context not initialized"
    chunk_start, chunk_end = chunk
    ctx = _worker_ctx
    result = compute_chunk_irradiance(
        geometry=ctx.geometry,
        sim_face_mask=ctx.sim_face_mask,
        solar_ts=ctx.solar_ts,
        hour_start=chunk_start,
        hour_end=chunk_end,
        attenuation_model=ctx.attenuation_model,
        face_to_group_id=ctx.face_to_group_id_full[ctx.sim_face_mask],
        ray_backend=ctx.ray_backend,
        ray_offset_m=ctx.ray_offset_m,
    )
    return chunk_start, chunk_end, result.irradiance_w_m2, result.shadow_mask


def run_simulation(config: PASAConfig) -> Path:
    """Run a full PASA-Sim simulation and write the result NPZ. Returns
    the path the NPZ was written to.
    """
    setup_logging(config.run.log_level, config.run.log_file)
    t_start = time.perf_counter()
    logger.info(f"PASA-Sim v{__version__} | mode={config.run.simulation_mode} | run_name={config.run.run_name}")

    ctx = build_simulation_context(config)
    geometry, solar_ts = ctx.geometry, ctx.solar_ts
    n_hours = solar_ts.n_hours
    n_sim_faces = int(ctx.sim_face_mask.sum())
    face_to_group_sim = ctx.face_to_group_id_full[ctx.sim_face_mask]

    all_chunks = chunk_ranges(n_hours, config.performance.hour_chunk_size)
    config_hash = hash_config(config.to_dict())
    checkpoint_mgr = CheckpointManager(config.resolved_checkpoint_dir(), config_hash)

    if config.run.force_restart:
        checkpoint_mgr.clear()
        checkpoint_state = None
    elif config.run.resume:
        checkpoint_state = checkpoint_mgr.load_if_valid()
    else:
        checkpoint_state = None

    save_irr = config.output.save_hourly_irradiance
    save_shadow = config.output.save_hourly_shadow_mask
    precision = np.float16 if config.performance.precision == "float16" else np.float32

    if checkpoint_state is not None:
        last_completed_hour = checkpoint_state.last_completed_hour
        annual_accum = checkpoint_state.arrays["annual_accum"].astype(np.float64)
        hourly_irradiance = checkpoint_state.arrays.get("hourly_irradiance") if save_irr else None
        hourly_shadow = checkpoint_state.arrays.get("hourly_shadow") if save_shadow else None
        if save_irr and hourly_irradiance is None:
            hourly_irradiance = np.zeros((n_sim_faces, n_hours), dtype=precision)
        if save_shadow and hourly_shadow is None:
            hourly_shadow = np.zeros((n_sim_faces, n_hours), dtype=bool)
    else:
        last_completed_hour = 0
        annual_accum = np.zeros(n_sim_faces, dtype=np.float64)
        hourly_irradiance = np.zeros((n_sim_faces, n_hours), dtype=precision) if save_irr else None
        hourly_shadow = np.zeros((n_sim_faces, n_hours), dtype=bool) if save_shadow else None

    remaining_chunks = [c for c in all_chunks if c[1] > last_completed_hour]

    if remaining_chunks:
        hours_since_checkpoint = 0
        pbar = tqdm(total=n_hours, initial=last_completed_hour, desc=f"{config.run.run_name} [{config.run.simulation_mode}]", unit="hr")

        def _consume(chunk_start: int, chunk_end: int, irr: np.ndarray, shadow: np.ndarray) -> None:
            nonlocal hours_since_checkpoint, last_completed_hour
            if save_irr:
                hourly_irradiance[:, chunk_start:chunk_end] = irr.astype(precision)
            if save_shadow:
                hourly_shadow[:, chunk_start:chunk_end] = shadow
            annual_accum[:] += irr.sum(axis=1)
            n_chunk_hours = chunk_end - chunk_start
            pbar.update(n_chunk_hours)
            hours_since_checkpoint += n_chunk_hours
            last_completed_hour = chunk_end
            if hours_since_checkpoint >= config.run.checkpoint_interval_hours:
                _save_checkpoint(checkpoint_mgr, last_completed_hour, annual_accum, hourly_irradiance, hourly_shadow)
                hours_since_checkpoint = 0

        if config.performance.n_workers > 1:
            logger.info(f"Multiprocessing enabled: {config.performance.n_workers} worker(s)")
            with ProcessPoolExecutor(max_workers=config.performance.n_workers, initializer=_worker_init, initargs=(config,)) as executor:
                # executor.map yields results in the SAME ORDER chunks were submitted, even
                # though workers may finish out of order -- this is what makes it safe to
                # checkpoint `last_completed_hour` monotonically without extra bookkeeping.
                for chunk_start, chunk_end, irr, shadow in executor.map(_worker_compute_chunk, remaining_chunks):
                    _consume(chunk_start, chunk_end, irr, shadow)
        else:
            for chunk_start, chunk_end in remaining_chunks:
                result = compute_chunk_irradiance(
                    geometry=geometry, sim_face_mask=ctx.sim_face_mask, solar_ts=solar_ts,
                    hour_start=chunk_start, hour_end=chunk_end,
                    attenuation_model=ctx.attenuation_model, face_to_group_id=face_to_group_sim,
                    ray_backend=ctx.ray_backend, ray_offset_m=ctx.ray_offset_m,
                )
                _consume(chunk_start, chunk_end, result.irradiance_w_m2, result.shadow_mask)

        pbar.close()
    else:
        logger.info("Checkpoint already covers the full simulation period; skipping straight to output assembly")

    result_path = _assemble_and_save(
        config, geometry, solar_ts, ctx, annual_accum, hourly_irradiance, hourly_shadow, precision, t_start,
    )
    checkpoint_mgr.clear()
    return result_path


def _save_checkpoint(
    mgr: CheckpointManager, last_completed_hour: int, annual_accum: np.ndarray,
    hourly_irradiance: np.ndarray | None, hourly_shadow: np.ndarray | None,
) -> None:
    arrays: dict[str, np.ndarray] = {"annual_accum": annual_accum.astype(np.float32)}
    if hourly_irradiance is not None:
        arrays["hourly_irradiance"] = hourly_irradiance
    if hourly_shadow is not None:
        arrays["hourly_shadow"] = hourly_shadow
    mgr.save(last_completed_hour, arrays)


def _assemble_and_save(
    config: PASAConfig, geometry: CityGeometry, solar_ts: SolarTimeSeries, ctx: SimulationContext,
    annual_accum: np.ndarray, hourly_irradiance: np.ndarray | None, hourly_shadow: np.ndarray | None,
    precision: type, t_start: float,
) -> Path:
    sim_mask = ctx.sim_face_mask
    n_faces = geometry.n_faces

    full_annual = np.full(n_faces, np.nan, dtype=np.float32)
    full_annual[sim_mask] = annual_accum.astype(np.float32)

    full_hourly_irr = None
    if hourly_irradiance is not None:
        full_hourly_irr = np.full((n_faces, solar_ts.n_hours), np.nan, dtype=precision)
        full_hourly_irr[sim_mask] = hourly_irradiance

    full_hourly_shadow = None
    if hourly_shadow is not None:
        # Excluded (non-simulated) faces are filled with False here as an arbitrary
        # placeholder -- they carry no meaning on their own. Any consumer of this array
        # (see analysis.py) must first restrict to `is_roof | is_wall` before
        # interpreting it, exactly as it must for `hourly_irradiance`'s NaN rows.
        full_hourly_shadow = np.zeros((n_faces, solar_ts.n_hours), dtype=bool)
        full_hourly_shadow[sim_mask] = hourly_shadow

    group_meta = ctx.attenuation_model.group_metadata()
    if group_meta:
        # Diagnostic full-year transmission per AOD group, for plotting/inspection.
        # Cheap: shape is (n_groups, n_hours), typically a few groups, not (n_faces, n_hours).
        group_meta = dict(group_meta)
        group_meta["aod_group_transmission"] = ctx.attenuation_model.compute_transmission(
            solar_ts.times, solar_ts.zenith_deg
        )

    aod_stats = None
    if "aod_group_value" in group_meta:
        vals = group_meta["aod_group_value"]
        aod_stats = {
            "mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)), "std": float(np.nanstd(vals)),
        }

    runtime_seconds = time.perf_counter() - t_start
    metadata = {
        "version": __version__,
        "created_utc": datetime.now(dt_timezone.utc).isoformat(),
        "simulation_mode": config.run.simulation_mode,
        "run_name": config.run.run_name,
        "city_glb_path": Path(config.city_model.glb_path).name,
        "year": config.time.year,
        "latitude": config.location.latitude,
        "longitude": config.location.longitude,
        "altitude_m": config.location.altitude_m,
        "timezone": config.location.timezone,
        "attenuation_model_name": ctx.attenuation_model.name,
        "aod_source_path": Path(config.pollution.aod.path).name if config.pollution.aod.path else None,
        "aod_statistics": aod_stats,
        "runtime_seconds": runtime_seconds,
        "config_snapshot": config.to_dict(),
        "git_commit": git_commit_hash(),
        "precision": config.performance.precision,
        "n_workers_used": config.performance.n_workers,
        "ray_backend_used": ctx.ray_backend.name,
        "roof_tilt_threshold_deg": config.city_model.roof_tilt_threshold_deg,
        "wall_tilt_max_deg": config.city_model.wall_tilt_max_deg,
    }

    result = SimulationResult(
        face_id=geometry.face_id,
        building_id_code=geometry.building_id_code,
        is_roof=geometry.is_roof,
        is_wall=geometry.is_wall,
        area_m2=geometry.area_m2,
        normal=geometry.normal,
        centroid=geometry.centroid,
        tilt_deg=geometry.tilt_deg,
        azimuth_deg=geometry.azimuth_deg,
        aod_group_id=ctx.face_to_group_id_full,
        annual_irradiance_wh_m2=full_annual,
        time_unix_s=solar_ts.times.tz_convert("UTC").as_unit("s").astype("int64").to_numpy(),
        solar_zenith_deg=solar_ts.zenith_deg,
        solar_azimuth_deg=solar_ts.azimuth_deg,
        solar_elevation_deg=solar_ts.elevation_deg,
        dni_clear_w_m2=solar_ts.dni_clear_w_m2,
        dni_extra_w_m2=solar_ts.dni_extra_w_m2,
        hourly_irradiance_w_m2=full_hourly_irr,
        hourly_shadow_mask=full_hourly_shadow,
        aod_group_transmission=group_meta.get("aod_group_transmission"),
        aod_group_value=group_meta.get("aod_group_value"),
        aod_group_lat=group_meta.get("aod_group_lat"),
        aod_group_lon=group_meta.get("aod_group_lon"),
        building_id_lookup=geometry.building_id_lookup,
        metadata=metadata,
    )

    output_path = config.resolved_output_dir() / f"{config.run.run_name}.npz"
    saved_path = save_simulation_result(result, output_path)
    logger.info(f"Simulation complete in {runtime_seconds:.1f}s -> {saved_path}")
    return saved_path
