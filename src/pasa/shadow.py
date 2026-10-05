"""
shadow.py

Ray-traced direct-beam visibility: for a batch of (face, hour) candidates,
determine whether the straight line from that face toward the sun is
blocked by any other part of the city mesh.

Performance design
--------------------
Two culls happen *before* any ray is cast (in irradiance.py, using the
candidate_mask this module receives): faces where the sun is behind the
surface (cos(incidence) <= 0) and hours where the sun is below the horizon
never generate a ray at all -- there is no geometric way they could be
lit, so testing them against the mesh would be pure waste. What remains is
flattened across the whole (face, hour) chunk into one batch of rays and
handed to trimesh's vectorized ``intersects_any`` in a single call, rather
than looping in Python per hour or per face. Embree (via the `embreex`
package) is used automatically when available and enabled in config, with
a transparent fallback to trimesh's pure-NumPy triangle intersector
otherwise -- the calling code never needs to know which backend is active.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .geometry import CityGeometry
from .utilities import get_logger

logger = get_logger()


@dataclass
class RayBackend:
    intersector: object
    name: str  # "embree" | "triangle"


def build_ray_backend(geometry: CityGeometry, raw_faces_mesh: trimesh.Trimesh, use_embree_if_available: bool) -> RayBackend:
    """Construct the fastest available ray intersector for the given mesh.

    Parameters
    ----------
    raw_faces_mesh:
        The full triangle mesh (including excluded faces -- undersides and
        neighbouring buildings can still occlude a simulated face's view
        of the sun even if they aren't themselves simulated).
    """
    if use_embree_if_available and trimesh.ray.has_embree:
        from trimesh.ray.ray_pyembree import RayMeshIntersector as EmbreeIntersector

        logger.info("Ray backend: Embree (embreex)")
        return RayBackend(intersector=EmbreeIntersector(raw_faces_mesh), name="embree")

    if use_embree_if_available:
        logger.warning(
            "performance.use_embree_if_available=True but no Embree backend is installed "
            "(pip install embreex); falling back to trimesh's pure-NumPy ray/triangle "
            "intersector. This is correct but noticeably slower for large scenes/long runs."
        )
    from trimesh.ray.ray_triangle import RayMeshIntersector as TriangleIntersector

    logger.info("Ray backend: triangle (pure NumPy fallback)")
    return RayBackend(intersector=TriangleIntersector(raw_faces_mesh), name="triangle")


def compute_visibility(
    backend: RayBackend,
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    sun_vectors_chunk: np.ndarray,
    candidate_mask: np.ndarray,
    ray_offset_m: float,
) -> np.ndarray:
    """Ray-traced sun visibility for every (face, hour) pair flagged True
    in `candidate_mask`. Pairs flagged False are returned as not-visible
    (False) without ever generating a ray -- the caller has already
    determined they cannot be lit (self-shadowed or sun below horizon).

    Parameters
    ----------
    face_centroids, face_normals:
        (n_faces, 3) arrays for the *simulated* faces (roof + wall only).
    sun_vectors_chunk:
        (chunk_hours, 3) unit vectors, one per hour in this chunk.
    candidate_mask:
        (n_faces, chunk_hours) bool; True where cos(incidence) > 0 and the
        sun is above the horizon for that (face, hour) pair.
    ray_offset_m:
        Distance to nudge each ray's origin along its face normal before
        casting, to avoid the ray immediately re-intersecting its own
        source triangle. Should already be scaled to the mesh's size by
        the caller (see geometry.CityGeometry.bbox_diagonal_m and
        config.CityModelConfig.ray_offset_scale).

    Returns
    -------
    (n_faces, chunk_hours) bool array: True where the face is actually
    illuminated (candidate AND not occluded by any other triangle).
    """
    n_faces, chunk_hours = candidate_mask.shape
    visible = np.zeros((n_faces, chunk_hours), dtype=bool)

    face_idx, hour_idx = np.nonzero(candidate_mask)
    if face_idx.size == 0:
        return visible

    origins = face_centroids[face_idx] + face_normals[face_idx] * ray_offset_m
    directions = sun_vectors_chunk[hour_idx]

    hit = backend.intersector.intersects_any(ray_origins=origins, ray_directions=directions)
    visible[face_idx, hour_idx] = ~hit
    return visible
