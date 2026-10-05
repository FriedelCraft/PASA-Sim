"""
irradiance.py

The physics core: for one hour-chunk, combine face geometry, solar
position, ray-traced shadow visibility, and whatever attenuation model was
selected (models.ClearSkyModel or models.AODAttenuationModel -- this
module never checks which) into direct-beam irradiance per simulated face
per hour.

    I_face(t) = DNI_clear(t) * transmission_group(t) * cos(theta_face(t)) * visible_face(t)

This is a direct-beam-only model (no diffuse-sky or ground-reflected
component), matching the scope of the pre-upgrade baseline pipeline this
framework replaces, so clear-vs-pollution comparisons stay physically
apples-to-apples with previously validated results.

The two culls that make ray tracing tractable both happen here, before
shadow.compute_visibility is ever called:
  1. sun below the horizon for the whole hour -> skip ray tracing entirely
  2. cos(incidence) <= 0 for a given (face, hour) -> self-shadowed, no ray
     needed (a face never needs to check whether *something else* blocks
     the sun once its own geometry already rules it out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CityGeometry
from .models.base import BaseAttenuationModel
from .shadow import RayBackend, compute_visibility
from .solar import SolarTimeSeries
from .utilities import get_logger

logger = get_logger()


@dataclass
class ChunkResult:
    irradiance_w_m2: np.ndarray  # (n_sim_faces, chunk_hours) float32
    shadow_mask: np.ndarray  # (n_sim_faces, chunk_hours) bool, True == shadowed/no direct sun


def compute_chunk_irradiance(
    geometry: CityGeometry,
    sim_face_mask: np.ndarray,
    solar_ts: SolarTimeSeries,
    hour_start: int,
    hour_end: int,
    attenuation_model: BaseAttenuationModel,
    face_to_group_id: np.ndarray,
    ray_backend: RayBackend,
    ray_offset_m: float,
) -> ChunkResult:
    """Compute direct-beam irradiance for simulated faces over
    ``[hour_start, hour_end)`` of ``solar_ts``.

    Parameters
    ----------
    sim_face_mask:
        (n_faces,) bool, True for faces that participate in the
        simulation (``geometry.simulated_mask()`` -- roof or wall, not
        excluded). Precomputed once by the caller and reused across every
        chunk rather than recomputed per call.
    face_to_group_id:
        (n_sim_faces,) int, already subset to the simulated faces, mapping
        each to a row of the attenuation model's transmission output.
    """
    normals = geometry.normal[sim_face_mask]
    centroids = geometry.centroid[sim_face_mask]
    n_sim_faces = normals.shape[0]
    chunk_hours = hour_end - hour_start

    sun_vectors = solar_ts.sun_vector_enu[hour_start:hour_end]  # (chunk_hours, 3)
    dni_clear = solar_ts.dni_clear_w_m2[hour_start:hour_end]  # (chunk_hours,)
    elevation = solar_ts.elevation_deg[hour_start:hour_end]  # (chunk_hours,)
    zenith_chunk = solar_ts.zenith_deg[hour_start:hour_end]
    times_chunk = solar_ts.times[hour_start:hour_end]

    cos_theta = normals @ sun_vectors.T  # (n_sim_faces, chunk_hours)

    sun_up = elevation > 0.0  # (chunk_hours,)
    candidate_mask = (cos_theta > 0.0) & sun_up[None, :]

    if not candidate_mask.any():
        # Whole chunk is night, or geometrically impossible for every face (shouldn't
        # happen in practice, but is a perfectly valid degenerate chunk) -- skip ray
        # tracing and the attenuation-model call entirely.
        zeros = np.zeros((n_sim_faces, chunk_hours), dtype=np.float32)
        return ChunkResult(irradiance_w_m2=zeros, shadow_mask=np.ones_like(zeros, dtype=bool))

    visible = compute_visibility(
        backend=ray_backend,
        face_centroids=centroids,
        face_normals=normals,
        sun_vectors_chunk=sun_vectors,
        candidate_mask=candidate_mask,
        ray_offset_m=ray_offset_m,
    )

    transmission_by_group = attenuation_model.compute_transmission(times_chunk, zenith_chunk)  # (n_groups, chunk_hours)
    transmission = transmission_by_group[face_to_group_id, :]  # (n_sim_faces, chunk_hours)

    irradiance = dni_clear[None, :] * transmission * np.clip(cos_theta, 0.0, None) * visible
    irradiance = irradiance.astype(np.float32)

    shadow_mask = ~visible  # True where a face receives no direct sun this hour (night, self-shadow, or occluded)
    return ChunkResult(irradiance_w_m2=irradiance, shadow_mask=shadow_mask)
