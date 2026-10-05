#!/usr/bin/env python3
"""
reconstruct_glb_from_3dtiles.py

Reconstructs a single, complete GLB from a Cesium 3D Tiles (1.0/1.1)
dataset: a tileset.json plus the directory of tile GLBs it references.

    python reconstruct_glb_from_3dtiles.py \
        --tileset path/to/tileset.json \
        --output  merged_city.glb
    # (--tiles-dir defaults to tileset.json's own directory, the standard
    #  3D Tiles layout where content URIs are relative to it)

What it does
--------------
1. Recursively walks the tileset hierarchy from the root, accumulating
   each node's 4x4 `transform` (a node without one inherits its parent's
   accumulated transform unchanged -- per the 3D Tiles spec).
2. For every tile with `content`, loads that tile's GLB and further
   composes each geometry's *own* transform within that GLB's internal
   scene graph -- easy to miss, but tile-exporters routinely bake a
   Y-up/Z-up (or similar) conversion into the content GLB itself, on top
   of (not instead of) the tileset-level placement transform. Skipping
   this silently produces mirrored/rotated geometry.
3. Honors `refine`: under REFINE=REPLACE (the default, and the far more
   common case), a node's own content is skipped in favor of its
   children's higher-detail content covering the same area, to avoid
   duplicating overlapping LOD geometry in a "complete" reconstruction;
   under REFINE=ADD, both are kept. `refine` is inherited down the tree
   when a node doesn't specify its own, exactly like `transform`.
4. Follows external tilesets (`content.uri` pointing at another
   `*tileset*.json`) recursively, resolving relative URIs against that
   sub-tileset's own directory.
5. Merges every resulting geometry into one trimesh.Scene as a separate
   node (preserving each tile's own materials/vertex colors rather than
   flattening everything into one mesh, which would collapse multiple
   materials into one) with its fully composed world transform applied.
6. Computes the centroid (mean vertex position) of the fully merged,
   full-precision geometry and re-centers the *entire scene* about it in
   one shot (Scene.apply_transform, which folds the shift into every
   node's transform rather than mutating vertex data directly).
   Tileset-hierarchy transforms in 3D Tiles place geometry in ECEF
   (Earth-Centered, Earth-Fixed) meters -- typically 6-7 significant
   digits before the decimal point. glTF/GLB always stores POSITION as
   float32 (~7 significant decimal digits total), so exporting raw ECEF
   coordinates leaves near-zero precision for actual geometric detail.
   Re-centering first keeps exported coordinates small (usually within a
   few hundred/thousand meters of zero), leaving full float32 precision
   for the geometry itself.
7. Writes the recentered scene as one GLB, plus a sidecar JSON recording
   the centroid that was subtracted (so the output can be placed back in
   ECEF space later: world = local + centroid) and basic provenance.

Vertex positions, face indices, and (per-tile) materials are carried
through unmodified aside from the transform. Normals are (re)computed by
trimesh from the merged face geometry -- if a source tile GLB has no
authored NORMAL accessor (common for tools that only export POSITION +
indices), there is nothing to "preserve" for that tile; computing smooth
per-vertex normals from geometry is the standard, correct fallback so the
merged output still shades properly, and is applied uniformly whether or
not any individual input tile happened to author its own.

Coordinate system note: this script performs *re-centering* (a
translation) exactly as specified, not a re-orientation into a local
East-North-Up frame. The output's axes remain ECEF-aligned, just shifted
near the origin. If you need true local-ENU geometry (Z = up) for
something like a solar/shadow simulation, that needs an additional
rotation step (built from the centroid's geodetic latitude/longitude) on
top of this script's output -- ask if you want that added.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("reconstruct_glb")

DEFAULT_REFINE = "REPLACE"


def parse_3dtiles_transform(values: list[float] | None) -> np.ndarray:
    """Parse a 3D Tiles / glTF 16-number transform array into a 4x4
    matrix. These arrays are always COLUMN-MAJOR (3D Tiles spec
    'transform'; glTF spec node 'matrix') -- reshaping row-major (the
    numpy default) silently yields the transpose of the correct matrix,
    which is a well-known, easy-to-miss source of corrupted geometry.
    Returns the identity matrix if `values` is None (a node with no
    transform of its own inherits its parent's unchanged).
    """
    if values is None:
        return np.eye(4, dtype=np.float64)
    if len(values) != 16:
        raise ValueError(f"Expected a 16-element transform array, got {len(values)} elements")
    return np.asarray(values, dtype=np.float64).reshape(4, 4, order="F")


def ecef_to_geodetic(x: float, y: float, z: float, a: float = 6378137.0, f: float = 1.0 / 298.257223563) -> tuple[float, float]:
    """WGS84 ECEF (meters) -> geodetic (latitude, longitude) in degrees,
    via the closed-form Bowring (1976) approximation (sub-millimeter
    accurate, no iteration needed). Used ONLY to build the optional
    ECEF->ENU rotation for --enu -- it plays no part in, and cannot
    affect, the tileset/glTF transform composition that places geometry
    in ECEF in the first place (that composition is independently
    verified; see the module-level NOTES_ON_TRANSFORM_VERIFICATION
    docstring below).
    """
    e2 = f * (2 - f)
    b = a * (1 - f)
    ep2 = (a**2 - b**2) / b**2
    p = np.hypot(x, y)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)
    return float(np.degrees(lat)), float(np.degrees(lon))


def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """3x3 rotation whose rows are the East/North/Up unit vectors expressed
    in ECEF, at the given geodetic latitude/longitude -- the standard
    local-tangent-plane rotation. Applied to an ECEF *offset* vector
    (i.e. already recentered near the origin), it gives East/North/Up
    components, with Up matching gravity-vertical at that point --
    unlike raw (recentered) ECEF axes, which do not align with "up"
    anywhere except exactly at the North Pole.
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    return np.array([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])


def resolve_uri(base_dir: Path, uri: str) -> Path:
    """Resolve a tile/tileset-relative URI against `base_dir`, decoding
    percent-escapes (e.g. spaces as %20) and stripping any query string,
    while leaving absolute (`http(s)://`) URIs alone -- those aren't
    supported by this script (it's built for a local tileset directory)
    and are reported clearly rather than silently mishandled.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        raise ValueError(f"Remote content URIs are not supported by this script: {uri}")
    return (base_dir / unquote(parsed.path)).resolve()


def load_tile_geometries(glb_path: Path) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    """Load one tile's GLB and return every (mesh, internal_world_transform)
    pair inside it. A tile GLB almost always contains exactly one
    geometry/node, but nothing in the 3D Tiles or glTF spec guarantees
    that, so every node in the file's own scene graph is handled, each
    with the transform it has *within that file* (composed down from
    that file's own root -- this is what trimesh's scene graph already
    resolves for us via `graph.get`).
    """
    loaded = trimesh.load(str(glb_path), process=False, force="scene")
    results: list[tuple[trimesh.Trimesh, np.ndarray]] = []

    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph.get(node_name)
            mesh = loaded.geometry.get(geom_name)
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
                results.append((mesh, transform))
    elif isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
        results.append((loaded, np.eye(4)))

    if not results:
        logger.warning(f"No triangle geometry found in {glb_path.name}")
    return results


def traverse_tileset_node(
    node: dict[str, Any],
    base_dir: Path,
    parent_transform: np.ndarray,
    parent_refine: str,
    scene: trimesh.Scene,
    stats: dict[str, int],
    used_node_names: set[str],
) -> None:
    """Recursively walk one tileset node and its children, adding every
    tile's geometry to `scene` with its fully composed world transform.
    """
    world_transform = parent_transform @ parse_3dtiles_transform(node.get("transform"))
    refine = node.get("refine", parent_refine).upper()
    children = node.get("children", [])
    content = node.get("content")

    # REFINE=REPLACE: children (if any) supersede this node's own content at
    # higher detail over the same area -- including both would duplicate
    # geometry in a "complete" reconstruction. REFINE=ADD: children
    # supplement rather than replace, so both are kept.
    use_own_content = content is not None and (refine == "ADD" or not children)

    if use_own_content:
        uri = content.get("uri") or content.get("url")  # "url" = legacy pre-1.0 key
        if not uri:
            logger.warning("Tile has a 'content' object with no uri/url; skipping")
        else:
            content_path = resolve_uri(base_dir, uri)
            if content_path.suffix.lower() == ".json" or "tileset" in content_path.name.lower():
                _load_external_tileset(content_path, world_transform, refine, scene, stats, used_node_names)
            elif not content_path.exists():
                logger.warning(f"Referenced content not found, skipping: {content_path}")
            else:
                _add_tile_content(content_path, world_transform, scene, stats, used_node_names)

    for child in children:
        traverse_tileset_node(child, base_dir, world_transform, refine, scene, stats, used_node_names)


def _add_tile_content(
    glb_path: Path, world_transform: np.ndarray, scene: trimesh.Scene, stats: dict[str, int], used_node_names: set[str],
) -> None:
    try:
        geometries = load_tile_geometries(glb_path)
    except Exception as exc:  # noqa: BLE001 - one bad tile must not abort the whole reconstruction
        logger.warning(f"Failed to load {glb_path.name}: {exc}")
        stats["tiles_failed"] += 1
        return

    for i, (mesh, internal_transform) in enumerate(geometries):
        final_transform = world_transform @ internal_transform
        base_name = glb_path.stem
        node_name = base_name if i == 0 else f"{base_name}_{i}"
        while node_name in used_node_names:  # guard against duplicate stems across sub-tilesets
            node_name = f"{node_name}_dup"
        used_node_names.add(node_name)

        scene.add_geometry(mesh, node_name=node_name, geom_name=node_name, transform=final_transform)
        stats["tiles_loaded"] += 1
        stats["faces"] += len(mesh.faces)
        stats["vertices"] += len(mesh.vertices)


def _load_external_tileset(
    tileset_path: Path, parent_transform: np.ndarray, parent_refine: str,
    scene: trimesh.Scene, stats: dict[str, int], used_node_names: set[str],
) -> None:
    if not tileset_path.exists():
        logger.warning(f"External tileset not found, skipping: {tileset_path}")
        return
    logger.info(f"Following external tileset: {tileset_path}")
    with open(tileset_path) as f:
        sub_tileset = json.load(f)
    traverse_tileset_node(sub_tileset["root"], tileset_path.parent, parent_transform, parent_refine, scene, stats, used_node_names)


def reconstruct(tileset_path: Path, tiles_dir: Path, output_path: Path, enu: bool = False) -> dict[str, Any]:
    t_start = time.perf_counter()
    logger.info(f"Loading tileset: {tileset_path}")
    with open(tileset_path) as f:
        tileset = json.load(f)

    asset_version = tileset.get("asset", {}).get("version", "unknown")
    logger.info(f"3D Tiles asset version: {asset_version}")

    scene = trimesh.Scene()
    stats: dict[str, int] = {"tiles_loaded": 0, "tiles_failed": 0, "faces": 0, "vertices": 0}
    used_node_names: set[str] = set()

    traverse_tileset_node(
        tileset["root"], tiles_dir, np.eye(4), tileset["root"].get("refine", DEFAULT_REFINE),
        scene, stats, used_node_names,
    )

    if stats["tiles_loaded"] == 0:
        raise RuntimeError(
            "No tile geometry was loaded -- check that --tiles-dir points at the directory "
            "containing the GLB files referenced by the tileset's content.uri values."
        )
    logger.info(
        f"Merged {stats['tiles_loaded']} tile(s) ({stats['tiles_failed']} failed): "
        f"{stats['vertices']:,} vertices, {stats['faces']:,} faces (pre-recenter, full ECEF scale)"
    )

    merged_preview = scene.to_geometry()  # throwaway single-mesh view, used only to compute the centroid
    centroid = merged_preview.vertices.mean(axis=0)
    logger.info(f"Global centroid (ECEF meters): [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

    recenter = np.eye(4)
    recenter[:3, 3] = -centroid

    enu_info: dict[str, Any] | None = None
    if enu:
        # Recenter first (translation), THEN rotate ECEF axes into local East-North-Up at
        # the centroid's geodetic position -- this is an ADDITIONAL rotation on top of the
        # already-verified recentering translation, not a replacement for it. Without this,
        # "up" in the exported file is a combination of ECEF X/Y/Z that varies by location
        # on the globe and does not point in any single fixed direction -- correct for
        # geospatial placement, but disorienting in a Z-up viewer (Blender) or a viewer
        # that isn't given the geodetic context to place ECEF-aligned geometry (a raw GLB
        # dropped into Cesium without re-georeferencing it).
        lat_deg, lon_deg = ecef_to_geodetic(*centroid)
        R_enu = ecef_to_enu_rotation(lat_deg, lon_deg)
        rotate = np.eye(4)
        rotate[:3, :3] = R_enu
        combined = rotate @ recenter
        scene.apply_transform(combined)
        enu_info = {"centroid_latitude_deg": lat_deg, "centroid_longitude_deg": lon_deg, "ecef_to_enu_rotation": R_enu.tolist()}
        logger.info(f"Applied ECEF->ENU rotation at centroid geodetic position: lat={lat_deg:.6f}, lon={lon_deg:.6f}")
    else:
        scene.apply_transform(recenter)

    bounds = scene.bounds
    logger.info(f"Post-recenter bounds (meters, relative to centroid): min={bounds[0].round(2)}, max={bounds[1].round(2)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # include_normals=True forces trimesh to compute (or pass through, if a source tile
    # happened to author its own) per-vertex normals for every mesh at export time.
    # Without it, trimesh's GLB exporter only writes a NORMAL accessor for meshes that
    # already had .vertex_normals cached before export -- freshly loaded tile meshes
    # don't, so omitting this flag silently drops normals from the whole output.
    scene.export(str(output_path), file_type="glb", include_normals=True)

    runtime_s = time.perf_counter() - t_start
    sidecar = {
        "centroid_ecef_m": centroid.tolist(),
        "enu_applied": enu,
        "note": (
            "Recentered (world = local + centroid_ecef_m) AND rotated into local "
            "East-North-Up (Z=up) at the centroid's geodetic position -- see "
            "centroid_latitude_deg/centroid_longitude_deg/ecef_to_enu_rotation below."
            if enu else
            "This is a re-centering translation only (world = local + centroid_ecef_m), "
            "not a rotation into a local East-North-Up frame. Axes remain ECEF-aligned -- "
            "re-run with --enu if you need Z=up geometry for a standard (non-georeferencing) viewer."
        ),
        "source_tileset": str(tileset_path),
        "source_asset_version": asset_version,
        "tiles_merged": stats["tiles_loaded"],
        "tiles_failed": stats["tiles_failed"],
        "total_vertices": stats["vertices"],
        "total_faces": stats["faces"],
        "bounds_local_m": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "output_glb": str(output_path),
        "runtime_seconds": round(runtime_s, 2),
    }
    if enu_info:
        sidecar.update(enu_info)
    sidecar_path = output_path.with_suffix("").with_suffix(".centroid.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"Wrote {output_path} and {sidecar_path} in {runtime_s:.1f}s")
    return sidecar


def collect_leaf_content(
    node: dict[str, Any], base_dir: Path, parent_transform: np.ndarray, parent_refine: str, out: list[tuple[Path, np.ndarray]],
) -> None:
    """Lightweight traversal mirroring traverse_tileset_node's exact same
    transform-accumulation and REFINE logic, but only collecting
    (content_path, world_transform) pairs in visitation order -- no mesh
    loading, no scene building. This is a separate, read-only function so
    --debug mode never touches (or risks changing the behavior of) the
    actual reconstruction traversal above.
    """
    world_transform = parent_transform @ parse_3dtiles_transform(node.get("transform"))
    refine = node.get("refine", parent_refine).upper()
    children = node.get("children", [])
    content = node.get("content")
    use_own_content = content is not None and (refine == "ADD" or not children)

    if use_own_content:
        uri = content.get("uri") or content.get("url")
        if uri:
            content_path = resolve_uri(base_dir, uri)
            if content_path.suffix.lower() == ".json" or "tileset" in content_path.name.lower():
                if content_path.exists():
                    with open(content_path) as f:
                        sub_tileset = json.load(f)
                    collect_leaf_content(sub_tileset["root"], content_path.parent, world_transform, refine, out)
            elif content_path.exists():
                out.append((content_path, world_transform))

    for child in children:
        collect_leaf_content(child, base_dir, world_transform, refine, out)


def run_debug_mode(tileset_path: Path, tiles_dir: Path, output_path: Path) -> None:
    """Export ONE tile, then TWO (traversal-order-adjacent) tiles, printing
    the full raw-bbox -> tileset-transform -> internal-transform ->
    composed-transform -> transformed-bbox trail for each, so the two-tile
    output's adjacency can be checked by hand (numerically here, and
    visually in Blender/Cesium) before committing to the full merge.
    """
    with open(tileset_path) as f:
        tileset = json.load(f)
    leaves: list[tuple[Path, np.ndarray]] = []
    collect_leaf_content(tileset["root"], tiles_dir, np.eye(4), tileset["root"].get("refine", DEFAULT_REFINE), leaves)
    if not leaves:
        raise RuntimeError("No tile content found while collecting leaves for --debug")
    logger.info(f"--debug: {len(leaves)} leaf tile(s) found; inspecting the first 1 and first 2")

    def _inspect_and_export(subset: list[tuple[Path, np.ndarray]], out_path: Path) -> np.ndarray:
        scene = trimesh.Scene()
        world_centroids = []
        for content_path, world_tf in subset:
            for mesh, internal_tf in load_tile_geometries(content_path):
                final_tf = world_tf @ internal_tf
                verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
                world_verts = (final_tf @ verts_h.T).T[:, :3]
                world_centroids.append(world_verts.mean(axis=0))

                logger.info(f"  [{content_path.name}]")
                logger.info(f"    1. raw local bbox      min={mesh.vertices.min(0).round(3)} max={mesh.vertices.max(0).round(3)}")
                logger.info(f"    2. tileset transform\n{world_tf}")
                logger.info(f"    3. internal transform\n{internal_tf}")
                logger.info(f"    4. composed transform\n{final_tf}")
                logger.info(f"    5. transformed bbox    min={world_verts.min(0).round(3)} max={world_verts.max(0).round(3)}")

                scene.add_geometry(mesh, node_name=content_path.stem, geom_name=content_path.stem, transform=final_tf)

        centroid = scene.to_geometry().vertices.mean(axis=0)
        recenter = np.eye(4)
        recenter[:3, 3] = -centroid
        scene.apply_transform(recenter)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(out_path), file_type="glb", include_normals=True)
        logger.info(f"  -> wrote {out_path} (recentered about this subset's own centroid)")
        return np.array(world_centroids)

    logger.info("=== DEBUG STEP 1: single tile ===")
    one_path = output_path.with_name(output_path.stem + "_debug_tile1.glb")
    _inspect_and_export(leaves[:1], one_path)

    logger.info("=== DEBUG STEP 2: two tiles (first two encountered in traversal order) ===")
    two_path = output_path.with_name(output_path.stem + "_debug_tile2.glb")
    centroids = _inspect_and_export(leaves[:2], two_path)
    if len(centroids) >= 2:
        dist = np.linalg.norm(centroids[0] - centroids[1])
        logger.info(f"  Centroid-to-centroid distance between the two tiles: {dist:.2f} m")
        logger.info("  Open both files in Blender/Cesium: the two buildings should sit near each "
                     "other with a plausible real-world gap, not overlap or fly apart. If they look "
                     "right, re-run without --debug for the full merge.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tileset", required=True, type=Path, help="Path to tileset.json")
    parser.add_argument("--tiles-dir", type=Path, default=None, help="Directory containing tile GLBs (default: tileset.json's own directory)")
    parser.add_argument("--output", required=True, type=Path, help="Output .glb path")
    parser.add_argument("--enu", action="store_true", help="Also rotate into local East-North-Up (Z=up) at the centroid, on top of recentering. See the module docstring's coordinate-system note.")
    parser.add_argument("--debug", action="store_true", help="Export one tile, then two neighboring tiles, with the full transform trail printed for each; skip the full merge.")
    args = parser.parse_args(argv)

    if not args.tileset.exists():
        parser.error(f"tileset.json not found: {args.tileset}")
    tiles_dir = args.tiles_dir or args.tileset.parent

    try:
        if args.debug:
            run_debug_mode(args.tileset, tiles_dir, args.output)
        else:
            reconstruct(args.tileset, tiles_dir, args.output, enu=args.enu)
    except Exception as exc:
        logger.error(f"Reconstruction failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
