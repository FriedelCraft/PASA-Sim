"""
ISRO Research Pipeline: 3D City Model Generator for Solar Irradiance Simulation
===============================================================================
Converts a 2D GeoPackage of building footprints with height attributes into a
watertight, metric, physically-based 3D GLB mesh.

Workflow:
1. Load GeoPackage and automatically project to a metric UTM CRS.
2. Clean and validate geometries (repair intersections, explode MultiPolygons).
3. Sanitize heights (handle missing/outliers using the dataset median).
4. Shift coordinates to a local origin (to avoid 32-bit float loss in raytracers).
5. Extrude watertight polygons using trimesh.
6. Post-process mesh (merge coincident vertices, remove degenerates, fix normals).
7. Validate physics readiness (watertight, manifold volume).
8. Export GLB and preview visualization.
"""

import os
import sys
import time
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.affinity import translate
import trimesh
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
INPUT_GPKG = "CIVIL_LINES_buildings_height.gpkg"
OUTPUT_GLB = "city.glb"
OUTPUT_PREVIEW = "building_height_preview.png"

# Height validation rules
MIN_HEIGHT = 1.0   # meters
MAX_HEIGHT = 120.0 # meters


def load_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Reads the GeoPackage and projects it to a metric UTM CRS if necessary.
    """
    print("Reading GeoPackage...")
    if not os.path.exists(filepath):
        print(f"ERROR: Cannot find {filepath}")
        sys.exit(1)

    gdf = gpd.read_file(filepath)
    if gdf.empty:
        print("ERROR: GeoPackage is empty.")
        sys.exit(1)

    print(f"  Loaded {len(gdf)} records.")
    print(f"  Original CRS: {gdf.crs}")

    # Project geographic coordinates to local UTM metric coordinates
    if gdf.crs and gdf.crs.is_geographic:
        print("  Geographic CRS detected. Estimating and projecting to local UTM...")
        utm_crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm_crs)
        print(f"  Projected to Metric CRS: {utm_crs}")
    else:
        print(f"  CRS is already projected (metric). Proceeding with {gdf.crs}.")

    return gdf


def clean_geometries(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict[str, int]]:
    """
    Repairs geometries and extracts explicit Polygons.
    Drops irrecoverable geometries.
    """
    print("Cleaning geometries...")
    stats = {"original": len(gdf), "repaired": 0, "skipped": 0, "valid_polygons_extracted": 0}
    
    clean_rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        
        if geom is None or geom.is_empty:
            stats["skipped"] += 1
            continue

        # Strip 3D Z-coordinates to prevent extrusion failures
        if geom.has_z:
            geom = shapely.force_2d(geom)

        is_repaired = False
        if not geom.is_valid:
            geom = make_valid(geom)
            is_repaired = True
            
        if not geom.is_valid:
            geom = geom.buffer(0)
            is_repaired = True
            
        if not geom.is_valid or geom.is_empty:
            stats["skipped"] += 1
            continue
            
        if is_repaired:
            stats["repaired"] += 1

        # Extract strict Polygons (ignore points/lines from GeometryCollections)
        polygons = []
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(list(geom.geoms))
        elif geom.geom_type == 'GeometryCollection':
            polygons.extend([g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))])

        if not polygons:
            stats["skipped"] += 1
            continue

        # Flatten MultiPolygons into separate rows for individual extrusion
        for poly in polygons:
            if isinstance(poly, MultiPolygon):
                for sub_poly in poly.geoms:
                    new_row = row.copy()
                    new_row.geometry = sub_poly
                    clean_rows.append(new_row)
                    stats["valid_polygons_extracted"] += 1
            else:
                new_row = row.copy()
                new_row.geometry = poly
                clean_rows.append(new_row)
                stats["valid_polygons_extracted"] += 1

    clean_gdf = gpd.GeoDataFrame(clean_rows, crs=gdf.crs)
    return clean_gdf, stats


def prepare_heights(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict[str, float]]:
    """
    Validates heights, computes the median, and fills invalid/missing heights.
    """
    print("Preparing building heights...")
    if 'height_m' not in gdf.columns:
        print("ERROR: 'height_m' column not found in GeoPackage.")
        sys.exit(1)

    # Coerce to numeric, turning text/errors into NaN
    gdf['height_m'] = pd.to_numeric(gdf['height_m'], errors='coerce')

    # Identify mathematically valid heights based on physical rules
    valid_mask = (gdf['height_m'] >= MIN_HEIGHT) & (gdf['height_m'] <= MAX_HEIGHT)
    valid_heights = gdf.loc[valid_mask, 'height_m']

    if valid_heights.empty:
        print("ERROR: No valid building heights found. Cannot compute median fallback.")
        sys.exit(1)

    median_h = valid_heights.median()
    avg_h = valid_heights.mean()
    
    # Identify invalid buildings
    invalid_mask = ~valid_mask | gdf['height_m'].isna()
    corrected_count = invalid_mask.sum()
    
    # Apply median correction
    gdf.loc[invalid_mask, 'height_m'] = median_h

    stats = {
        "corrected_buildings": corrected_count,
        "median_height": median_h,
        "average_height": avg_h,
        "min_valid": valid_heights.min(),
        "max_valid": valid_heights.max()
    }
    
    print(f"  Number of corrected buildings: {corrected_count}")
    print(f"  Median height used for corrections: {median_h:.2f} m")

    return gdf, stats


def extrude_buildings(gdf: gpd.GeoDataFrame) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Translates polygons to local origin and extrudes them into 3D watertight meshes.
    Merges everything into a single validated mesh.
    """
    print("Extruding buildings...")
    
    # 1. Calculate centroid and shift to local origin
    # Ray tracers use 32-bit floats. Large UTM coordinates (e.g., X=500000, Y=4000000) 
    # cause vertices to snap to grids, ruining normals and watertightness.
    bounds = gdf.total_bounds
    origin_x = (bounds[0] + bounds[2]) / 2.0
    origin_y = (bounds[1] + bounds[3]) / 2.0
    
    print(f"  Shifting geometries to local origin (X: {origin_x:.2f}, Y: {origin_y:.2f})")
    
    building_meshes = []
    extrusion_fails = 0

    # 2. Iterate and Extrude
    for i, (_, row) in enumerate(gdf.iterrows(), 1):

        geom = row.geometry
        height = float(row['height_m'])
        
        # Translate locally
        local_geom = translate(geom, xoff=-origin_x, yoff=-origin_y)
        if i % 100 == 0:
            print(f"  Extruded {i}/{len(gdf)} buildings...")
        try:
            mesh = trimesh.creation.extrude_polygon(local_geom, height=height)

            # Skip invalid meshes
            if not mesh.is_volume:
                extrusion_fails += 1
                continue

            building_meshes.append(mesh)

        except Exception as e:
            print(f"Building {i}: {e}")
            extrusion_fails += 1
            continue

    if not building_meshes:
        print("ERROR: All building extrusions failed.")
        sys.exit(1)

    print("Generating city mesh (concatenating and optimizing)...")
    # 3. Concatenate all geometries into a single mesh object
    city_mesh = trimesh.util.concatenate(building_meshes)
    city_mesh = trimesh.Trimesh(
        vertices=city_mesh.vertices,
        faces=city_mesh.faces,
        process=False
    )
    

    # 4. Critical Post-Processing for Ray Tracing Physics
    print("  Merging coincident vertices...")
    city_mesh.merge_vertices(merge_tex=False, merge_norm=False)
    
    print("  Removing degenerate faces...")
    # Remove degenerate faces
    city_mesh.update_faces(city_mesh.nondegenerate_faces())

    # Remove duplicate faces (if available)
    try:
        city_mesh.update_faces(city_mesh.unique_faces())
    except AttributeError:
        pass
    
    print("  Fixing inverted normals...")
    city_mesh.fix_normals()

    stats = {
        "meshes_concatenated": len(building_meshes),
        "extrusion_failures": extrusion_fails,
        "origin_x": origin_x,
        "origin_y": origin_y
    }
    
    return city_mesh, stats


def validate_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Checks the physical properties of the generated mesh.
    """
    print("Checking mesh quality...")
    
    is_watertight = mesh.is_watertight
    is_volume = mesh.is_volume
    
    if not is_watertight:
        print("  WARNING: Mesh is NOT completely watertight. Ray tracing shadows may bleed.")
    if not is_volume:
        print("  WARNING: Mesh is NOT a valid manifold volume. Check for self-intersections.")
        
    stats = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounding_box": mesh.bounds,
        "is_watertight": is_watertight,
        "is_volume": is_volume
    }
    return stats


def export_glb(mesh: trimesh.Trimesh, filepath: str) -> float:
    """
    Exports the mesh to GLB format and returns file size in MB.
    """
    print("Exporting GLB...")
    mesh.export(filepath, file_type='glb')
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved {filepath} ({file_size_mb:.2f} MB)")
    return file_size_mb


def generate_preview(gdf: gpd.GeoDataFrame, filepath: str):
    """
    Generates a 2D choropleth map of the building heights.
    """
    print("Generating height preview visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot using a warm colormap suitable for solar/height metrics
    gdf.plot(column='height_m', ax=ax, legend=True,
             cmap='inferno', edgecolor='black', linewidth=0.2,
             legend_kwds={'label': "Building Height (m)", 'shrink': 0.7})
             
    ax.set_title("3D City Model Extrusion Heights", fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved preview to {filepath}")


def main():
    start_time = time.time()
    print("="*60)
    print("URBAN SOLAR IRRADIANCE: 3D CITY MODEL GENERATOR")
    print("="*60)

    # 1. Load Data
    gdf = load_data(INPUT_GPKG)
    original_count = len(gdf)

    # 2. Visualize Original State
    generate_preview(gdf, OUTPUT_PREVIEW)

    # 3. Clean Geometry
    clean_gdf, geom_stats = clean_geometries(gdf)

    # 4. Prepare Heights
    ready_gdf, height_stats = prepare_heights(clean_gdf)

    # 5. Extrude and Process Mesh
    city_mesh, mesh_stats = extrude_buildings(ready_gdf)

    # 6. Validate
    val_stats = validate_mesh(city_mesh)

    # 7. Export
    glb_size = export_glb(city_mesh, OUTPUT_GLB)

    # 8. Summary & Checklist
    processing_time = time.time() - start_time
    bounds = val_stats["bounding_box"]
    dim_x = bounds[1][0] - bounds[0][0]
    dim_y = bounds[1][1] - bounds[0][1]
    max_h = bounds[1][2] - bounds[0][2]

    print("\n" + "-"*40)
    print("PROCESSING SUMMARY")
    print("-"*40)
    print(f"Input buildings processed : {original_count}")
    print(f"Buildings repaired        : {geom_stats['repaired']}")
    print(f"Buildings skipped         : {geom_stats['skipped']}")
    print(f"Extrusion failures        : {mesh_stats['extrusion_failures']}")
    print(f"Final distinct polygons   : {mesh_stats['meshes_concatenated']}")
    print("-" * 40)
    print(f"Average height            : {height_stats['average_height']:.2f} m")
    print(f"Median height             : {height_stats['median_height']:.2f} m")
    print("-" * 40)
    print(f"GLB file size             : {glb_size:.2f} MB")
    print(f"Vertices                  : {val_stats['vertices']:,}")
    print(f"Faces                     : {val_stats['faces']:,}")
    print(f"Bounding box dimensions   : {dim_x:.1f}m(L) x {dim_y:.1f}m(W) x {max_h:.1f}m(H)")
    print(f"Processing time           : {processing_time:.1f} seconds")
    print("-" * 40)

    print("\n==========================================")
    print("VALIDATION CHECKLIST FOR SOLAR SIMULATION")
    print("==========================================")
    print(f"[ {'✓' if val_stats['is_watertight'] else 'X'} ] Watertight Mesh (No holes for light leaks)")
    print(f"[ {'✓' if val_stats['is_volume'] else 'X'} ] Valid Manifold Volume (Proper winding orders)")
    print(f"[ ✓ ] Outward-facing Normals Repaired")
    print(f"[ ✓ ] Local Metric Translation Applied (X: {mesh_stats['origin_x']:.1f}, Y: {mesh_stats['origin_y']:.1f})")
    print(f"[ ✓ ] Coincident Vertices Merged")
    print("==========================================\n")
    print("Done. The GLB file is ready to be loaded via `trimesh.load()`.")


if __name__ == "__main__":
    main()