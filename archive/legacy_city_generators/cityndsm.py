import json
import time
import logging
from typing import List, Tuple, Optional, Any, Union

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import trimesh
import shapely # NEW: Imported to handle 3D coordinate stripping
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from shapely.affinity import translate
from tqdm import tqdm

# Configure basic logging to suppress noisy external library warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_polygons(geom: BaseGeometry) -> List[Polygon]:
    """
    Recursively extracts only Polygon geometries from a given geometry,
    discarding Points, LineStrings, or empty geometries.
    """
    polygons = []
    if geom.is_empty:
        return polygons

    if isinstance(geom, Polygon):
        polygons.append(geom)
    elif isinstance(geom, MultiPolygon):
        polygons.extend(list(geom.geoms))
    elif geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            polygons.extend(extract_polygons(part))
            
    return polygons


def validate_and_clean_geometry(geom: Any) -> Optional[Union[Polygon, MultiPolygon]]:
    """
    Attempts to repair invalid geometries using make_valid() and buffer(0).
    Returns a valid Polygon, MultiPolygon, or None if unrecoverable.
    """
    if geom is None or not isinstance(geom, BaseGeometry) or geom.is_empty:
        return None
        
    # FIX: Strip Z coordinates to prevent trimesh triangulation crashes
    if geom.has_z:
        geom = shapely.force_2d(geom)

    if not geom.is_valid:
        try:
            geom = make_valid(geom)
        except Exception:
            pass

    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception:
            pass

    if not geom.is_valid or geom.is_empty:
        return None

    # Extract purely polygonal parts (make_valid can sometimes return Lines/Points)
    polys = extract_polygons(geom)
    if not polys:
        return None

    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def get_ndsm_height(geom: BaseGeometry, src: rasterio.DatasetReader) -> float:
    """
    Masks the nDSM with the geometry and calculates the 90th percentile height.
    Ignores NoData and negative heights.
    """
    try:
        out_image, _ = mask(src, [geom], crop=True, filled=False)
        nodata = src.nodata
        
        # Flatten and filter out invalid values
        pixels = out_image.flatten()
        if nodata is not None:
            valid_pixels = pixels[(pixels != nodata) & (pixels > 0)]
        else:
            valid_pixels = pixels[pixels > 0]
            
        if valid_pixels.size == 0:
            return 0.0
            
        return float(np.percentile(valid_pixels, 90))
    except Exception:
        # Gracefully fail if geometry is out of raster bounds or masking fails
        return 0.0


def get_building_height(row: gpd.GeoSeries, src: rasterio.DatasetReader) -> Tuple[float, str]:
    """
    Determines building height with nDSM as the PRIMARY source.
    FALLBACK 1: 'height_m' attribute.
    FALLBACK 2: 3.0 meters default.
    """
    # 1. PRIMARY: Extract from nDSM
    h_ndsm = get_ndsm_height(row['geometry'], src)
    
    # Requirement: use nDSM if valid and >= 1.0m
    if h_ndsm >= 1.0:
        return h_ndsm, 'ndsm'

    # 2. FALLBACK: Try attribute height if nDSM failed or was < 1m
    try:
        if 'height_m' in row and row['height_m'] is not None:
            h_attr = float(row['height_m'])
            if not np.isnan(h_attr) and h_attr >= 1.0:
                return h_attr, 'attribute'
    except (ValueError, TypeError, KeyError):
        pass # Fall through to default

    # 3. DEFAULT: If both nDSM and attribute fail
    return 3.0, 'default'


def extrude_building(geom: Union[Polygon, MultiPolygon], height: float, exceptions_list: List[Exception]) -> List[trimesh.Trimesh]:
    """
    Extrudes Shapely polygons into 3D watertight meshes.
    """
    meshes = []
    polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    
    for poly in polygons:
        try:
            # trimesh.creation.extrude_polygon automatically triangulates top/bottom caps
            mesh = trimesh.creation.extrude_polygon(poly, height=height)
            meshes.append(mesh)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
            # FIX: Print the first 5 exceptions instead of silently failing
            if len(exceptions_list) < 5:
                exceptions_list.append(e)
                logger.error(f"Extrusion failed: {e.__class__.__name__}: {e}")
            continue
            
    return meshes


def load_and_prepare_data(vector_path: str, raster_path: str) -> Tuple[gpd.GeoDataFrame, rasterio.DatasetReader]:
    """
    Loads vector and raster datasets. Reprojects vector if CRS mismatches.
    """
    print("Loading vector data...")
    gdf = gpd.read_file(vector_path)
    
    src = rasterio.open(raster_path)
    print(f"Loaded nDSM (CRS: {src.crs})")
    
    if gdf.crs != src.crs:
        print(f"Reprojecting vector geometries from {gdf.crs} to {src.crs}...")
        gdf = gdf.to_crs(src.crs)
        
    return gdf, src


def process_city_model(vector_path: str, ndsm_path: str, output_glb: str):
    """
    Main orchestration function with updated nDSM-primary logic and stats.
    """
    start_time = time.time()
    
    try:
        gdf, src = load_and_prepare_data(vector_path, ndsm_path)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # MODIFIED: Updated stats dictionary for new requirements
    stats = {
        'total': len(gdf),
        'valid': 0,
        'ndsm_height': 0,
        'attribute_height': 0,
        'default_height': 0,
        'skipped_invalid_geom': 0,
        'skipped_extrusion_fail': 0,
        'skipped_empty_mesh': 0,
        'successful_meshes': 0
    }
    
    # NEW: List to track successfully applied heights for Min/Max/Mean calculation
    applied_heights = []

    bounds = gdf.total_bounds
    center_x = (bounds[0] + bounds[2]) / 2.0
    center_y = (bounds[1] + bounds[3]) / 2.0
    
    origin_data = {
        "crs": str(gdf.crs),
        "origin_x": center_x,
        "origin_y": center_y
    }
    
    with open("city_origin.json", "w") as f:
        json.dump(origin_data, f, indent=4)
    print(f"Saved local coordinate origin to city_origin.json (X: {center_x:.2f}, Y: {center_y:.2f})")

    city_meshes = []
    extrusion_exceptions = []

    with src:
        print("Processing buildings...")
        for idx, row in tqdm(gdf.iterrows(), total=stats['total'], desc="Meshing City"):
            
            # 1. Validate & Clean
            valid_geom = validate_and_clean_geometry(row['geometry'])
            if not valid_geom:
                stats['skipped_invalid_geom'] += 1
                continue
                
            row['geometry'] = valid_geom 

            # 2. Extract Height (MODIFIED to track new sources)
            height, source = get_building_height(row, src)
            
            if source == 'ndsm':
                stats['ndsm_height'] += 1
            elif source == 'attribute':
                stats['attribute_height'] += 1
            else:
                stats['default_height'] += 1

            # 3. Shift to Local Coordinates
            local_geom = translate(valid_geom, xoff=-center_x, yoff=-center_y)

            # 4. Extrude
            building_meshes = extrude_building(local_geom, height, extrusion_exceptions)
            
            if not building_meshes:
                stats['skipped_extrusion_fail'] += 1
                continue
                
            if len(building_meshes) == 0:
                stats['skipped_empty_mesh'] += 1
                continue
                
            city_meshes.extend(building_meshes)
            
            # NEW: Record the height only if extrusion was successful
            applied_heights.append(height)
            
            stats['valid'] += 1
            stats['successful_meshes'] += len(building_meshes)

    duration = time.time() - start_time
    
    # NEW: Calculate aggregate height statistics
    mean_h = np.mean(applied_heights) if applied_heights else 0.0
    min_h = np.min(applied_heights) if applied_heights else 0.0
    max_h = np.max(applied_heights) if applied_heights else 0.0
    
    # MODIFIED: Print block updated with new requirements
    print("\n" + "="*30)
    print("PROCESSING SUMMARY")
    print("="*30)
    print(f"Processing Time        : {duration:.2f} seconds")
    print(f"Total buildings input  : {stats['total']}")
    print(f"Valid extruded groups  : {stats['valid']}")
    print(f"Successful meshes      : {stats['successful_meshes']}")
    print("-" * 30)
    print(f"Heights from nDSM      : {stats['ndsm_height']}")
    print(f"Heights from attribute : {stats['attribute_height']}")
    print(f"Heights from default   : {stats['default_height']}")
    print("-" * 30)
    print(f"Mean extracted height  : {mean_h:.2f} m")
    print(f"Min extracted height   : {min_h:.2f} m")
    print(f"Max extracted height   : {max_h:.2f} m")
    print("-" * 30)
    print(f"Skipped (Bad Geometry) : {stats['skipped_invalid_geom']}")
    print(f"Skipped (Meshing Fail) : {stats['skipped_extrusion_fail']}")
    print(f"Skipped (Empty Mesh)   : {stats['skipped_empty_mesh']}")
    print("="*30)

    if not city_meshes:
        print("ERROR: No valid buildings were generated. Aborting export.")
        return

    print("Concatenating meshes into a single scene...")
    final_mesh = trimesh.util.concatenate(city_meshes)
    
    print(f"Exporting watertight mesh to {output_glb}...")
    final_mesh.export(output_glb)


if __name__ == "__main__":
    # Define file paths here
    BUILDING_FILE = "buildings_aoi.gpkg"
    NDSM_FILE = "ndsm.tif"
    OUTPUT_GLB = "city.glb"
    
    process_city_model(BUILDING_FILE, NDSM_FILE, OUTPUT_GLB)