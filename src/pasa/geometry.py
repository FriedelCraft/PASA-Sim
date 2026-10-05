"""
geometry.py

Turns a RawCityMesh (io.mesh_io) into per-face geometric attributes:
outward unit normals, areas, centroids, tilt/azimuth angles, and the
roof / wall / excluded classification. This is the only module that knows
the mesh is assumed to be in a local ENU (East-North-Up, meters) frame
centered on the AOI, as already established by the upstream city-model /
georeferencing pipeline.

Angle conventions
------------------
- Tilt: angle from the vertical (+Z / Up) axis, in [0, 180] degrees.
  0 deg = horizontal face pointing straight up (flat roof).
  90 deg = vertical face (wall).
  >90 deg = face pointing at least partly downward (soffit / underside).
- Azimuth: compass bearing of the horizontal projection of the face
  normal, in [0, 360) degrees, measured clockwise from North -- the same
  convention pvlib uses for `surface_azimuth` (0=N, 90=E, 180=S, 270=W).
  Azimuth is undefined (returned as NaN) for perfectly horizontal faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .config import CityModelConfig
from .io.mesh_io import RawCityMesh
from .utilities import get_logger, wrap_angle_deg

logger = get_logger()


@dataclass
class CityGeometry:
    """All per-face geometric attributes the simulator needs, plus the
    building-id lookup table. Every array below is aligned on the same
    face axis (index i refers to the same triangle in every array).
    """

    face_id: np.ndarray  # (n,) int64, 0..n-1
    building_id_code: np.ndarray  # (n,) int32
    building_id_lookup: list[str]  # code -> original building name
    normal: np.ndarray  # (n, 3) float32, unit vectors, ENU frame
    centroid: np.ndarray  # (n, 3) float32, ENU meters
    area_m2: np.ndarray  # (n,) float32
    tilt_deg: np.ndarray  # (n,) float32
    azimuth_deg: np.ndarray  # (n,) float32, NaN where tilt == 0
    is_roof: np.ndarray  # (n,) bool
    is_wall: np.ndarray  # (n,) bool
    is_excluded: np.ndarray  # (n,) bool  (undersides / degenerate slivers)
    bbox_diagonal_m: float  # used to scale the ray-tracing self-intersection offset

    @property
    def n_faces(self) -> int:
        return int(self.face_id.shape[0])

    def simulated_mask(self) -> np.ndarray:
        """Boolean mask of faces that actually participate in the
        irradiance simulation (roof or wall, not excluded)."""
        return self.is_roof | self.is_wall


def _compute_tilt_azimuth(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized tilt/azimuth from unit ENU normals. See module docstring
    for the angle conventions.
    """
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    tilt_deg = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))).astype(np.float32)
    with np.errstate(invalid="ignore"):
        azimuth_deg = wrap_angle_deg(np.degrees(np.arctan2(nx, ny))).astype(np.float32)
    # Azimuth is undefined for a perfectly flat (nx == ny == 0) face.
    horizontal_norm = np.hypot(nx, ny)
    azimuth_deg = np.where(horizontal_norm < 1e-8, np.nan, azimuth_deg).astype(np.float32)
    return tilt_deg, azimuth_deg


def build_city_geometry(raw: RawCityMesh, cfg: CityModelConfig) -> CityGeometry:
    """Compute all derived per-face attributes and the roof/wall/excluded
    classification from a loaded RawCityMesh.

    Roof / wall / excluded logic
    -----------------------------
    - excluded: area below ``cfg.min_face_area_m2`` (degenerate slivers from
      mesh generation), OR tilt above ``cfg.wall_tilt_max_deg`` (downward-
      facing undersides/soffits, which cannot receive direct beam sunlight
      in an urban canyon and are out of scope for this direct-beam-only
      irradiance model).
    - roof: not excluded, tilt <= ``cfg.roof_tilt_threshold_deg``.
    - wall: not excluded, tilt in (``roof_tilt_threshold_deg``, ``wall_tilt_max_deg``].
    """
    mesh = trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces, process=False)

    normals = mesh.face_normals.astype(np.float32)
    areas = mesh.area_faces.astype(np.float32)
    centroids = mesh.triangles_center.astype(np.float32)
    bbox_diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    tilt_deg, azimuth_deg = _compute_tilt_azimuth(normals)

    is_excluded = (areas < cfg.min_face_area_m2) | (tilt_deg > cfg.wall_tilt_max_deg)
    is_roof = (~is_excluded) & (tilt_deg <= cfg.roof_tilt_threshold_deg)
    is_wall = (~is_excluded) & (~is_roof)

    n_faces = raw.n_faces
    n_excluded = int(is_excluded.sum())
    if n_excluded > 0:
        logger.info(
            f"Excluded {n_excluded:,}/{n_faces:,} faces "
            f"(degenerate area < {cfg.min_face_area_m2} m^2, or downward-facing tilt > {cfg.wall_tilt_max_deg} deg)"
        )
    logger.info(f"Classified {int(is_roof.sum()):,} roof faces, {int(is_wall.sum()):,} wall faces")

    return CityGeometry(
        face_id=np.arange(n_faces, dtype=np.int64),
        building_id_code=raw.face_building_id,
        building_id_lookup=list(raw.building_names),
        normal=normals,
        centroid=centroids,
        area_m2=areas,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        is_roof=is_roof,
        is_wall=is_wall,
        is_excluded=is_excluded,
        bbox_diagonal_m=bbox_diagonal,
    )
