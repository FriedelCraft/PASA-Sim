"""
io/mesh_io.py

Loads a GLB city model into world-space triangle geometry while preserving
per-building identity, and does nothing else -- derived geometric
attributes (normals, tilt, azimuth, roof/wall classification) live in
geometry.py. Keeping "read the file" and "interpret the geometry" separate
means a future OBJ/CityJSON/3D-Tiles loader only has to reimplement this
file, not geometry.py.

GLB layout assumption
----------------------
The loader expects the GLB to be a ``trimesh.Scene`` in which each building
is a separate named node (one mesh per building), consistent with the LOD1
GLB export convention already used upstream in the city-model generation
pipeline (each building = one scene node, node name = building identifier).
A GLB containing a single un-named merged mesh is also accepted as a
degenerate case with one synthetic building id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from ..utilities import get_logger

logger = get_logger()


@dataclass
class RawCityMesh:
    """World-space triangle soup with a per-face building id, prior to any
    normal/area/tilt/azimuth derivation.
    """

    vertices: np.ndarray  # (n_vertices, 3) float64, world/ENU meters
    faces: np.ndarray  # (n_faces, 3) int64, indices into `vertices`
    face_building_id: np.ndarray  # (n_faces,) int32, index into `building_names`
    building_names: list[str]  # building_id (int) -> original node name

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def load_raw_city_mesh(glb_path: str | Path) -> RawCityMesh:
    """Load a GLB file into a single world-space triangle soup with
    per-face building identity preserved.

    Raises
    ------
    FileNotFoundError
        If ``glb_path`` does not exist.
    ValueError
        If the file loads but contains zero triangles.
    """
    glb_path = Path(glb_path)
    if not glb_path.exists():
        raise FileNotFoundError(f"City model GLB not found: {glb_path}")

    logger.info(f"Loading city model: {glb_path}")
    loaded = trimesh.load(str(glb_path), force="scene", process=False)

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_building_id: list[np.ndarray] = []
    building_names: list[str] = []
    vertex_offset = 0

    if isinstance(loaded, trimesh.Scene) and len(loaded.graph.nodes_geometry) > 0:
        for building_idx, node_name in enumerate(loaded.graph.nodes_geometry):
            transform, geom_name = loaded.graph.get(node_name)
            mesh = loaded.geometry[geom_name]
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                continue
            world_vertices = trimesh.transformations.transform_points(mesh.vertices, transform)
            all_vertices.append(world_vertices)
            all_faces.append(mesh.faces + vertex_offset)
            all_building_id.append(np.full(len(mesh.faces), building_idx, dtype=np.int32))
            building_names.append(node_name)
            vertex_offset += len(world_vertices)
    else:
        # Degenerate case: a single, un-named mesh (no scene graph).
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"{glb_path} loaded but contains no triangles")
        logger.warning(
            "GLB has no named per-building scene nodes; treating the whole model as one building "
            "('building_0'). Building-level statistics in analysis.py will be degenerate."
        )
        all_vertices.append(mesh.vertices)
        all_faces.append(mesh.faces)
        all_building_id.append(np.zeros(len(mesh.faces), dtype=np.int32))
        building_names.append("building_0")

    if not all_faces:
        raise ValueError(f"{glb_path} loaded but contains no triangles")

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float64)
    faces = np.concatenate(all_faces, axis=0).astype(np.int64)
    face_building_id = np.concatenate(all_building_id, axis=0)

    logger.info(
        f"Loaded {len(building_names)} building(s), {faces.shape[0]:,} triangles, "
        f"{vertices.shape[0]:,} vertices"
    )
    return RawCityMesh(
        vertices=vertices,
        faces=faces,
        face_building_id=face_building_id,
        building_names=building_names,
    )
