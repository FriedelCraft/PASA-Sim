import json
import time
import logging
from typing import List, Tuple, Optional, Any, Union

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from shapely.affinity import translate
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_osm_numeric(val: Any) -> Optional[float]:
    """
    Safely parses OSM numeric tags. Handles edge cases like '2;3' (takes max),
    NaNs, strings, and missing values.
    """
    if val is None:
        return None
        
    # Handle float NaNs (which often appear in GeoPandas for missing string data)
    if isinstance(val, float) and np.isnan(val):
        return None

    try:
        if isinstance(val, str):
            # Sometimes OSM editors tag multiple levels e.g., "2;3"
            if ';' in val:
                return max(float(x.strip()) for x in val.split(';'))
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_polygons(geom: BaseGeometry) -> List[Polygon]:
    """Recursively extracts Polygon geometries, dropping Lines/Points."""
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
    Repairs geometries, strips 3D coordinates, and returns valid (Multi)Polygons.
    """
    if geom is None or not isinstance(geom, BaseGeometry) or geom.is_empty:
        return None
        
    # Strip Z coordinates to prevent trimesh triangulation crashes
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

    polys = extract_polygons(geom)
    if not polys:
        return None

    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def get_ndsm_height(geom: BaseGeometry, src: rasterio.DatasetReader) -> float:
    """Extracts the 90th percentile height from the nDSM."""
    try:
        out_image, _ = mask(src, [geom], crop=True, filled=False)
        nodata = src.nodata
        
        pixels = out_image.flatten()
        if nodata is not None:
            valid_pixels = pixels[(pixels != nodata) & (pixels > 0)]
        else:
            valid_pixels = pixels[pixels > 0]
            
        if valid_pixels.size == 0:
            return 0.0
            
        return float(np.percentile(valid_pixels, 90))
    except Exception:
        return 0.0


def get_building_height(row: gpd.GeoSeries, src: rasterio.DatasetReader) -> Tuple[float, str]:
    """
    Determines building height based on strictly ordered priorities:
    1. building:levels * 3.2m
    2. nDSM (90th percentile)
    3. Attribute fallback (height, height_m, etc.)
    4. Default 3.0m
    """
    # Priority 1: OSM building:levels
    if 'building:levels' in row:
        levels = parse_osm_numeric(row['building:levels'])
        if levels is not None and levels > 0:
            return levels * 3.2, 'levels'

    # Priority 2: nDSM Extraction
    h_ndsm = get_ndsm_height(row['geometry'], src)
    if h_ndsm >= 1.0:
        return h_ndsm, 'ndsm'

    # Priority 3: Numeric height attributes
    potential_attrs = ['height', 'height_m', 'roof_height', 'roof:height']
    for attr in potential_attrs:
        if attr in row:
            h_attr = parse_osm_numeric(row[attr])
            if h_attr is not None and h_attr >= 1.0:
                return h_attr, 'attribute'

    # Priority 4: Default fallback
    return 3.0, 'default'


def extrude_building(geom: Union[Polygon, MultiPolygon], height: float, exceptions_list: List[Exception]) -> List[trimesh.Trimesh]:
    """Extrudes polygons into watertight LOD1 block models."""
    meshes = []
    polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    
    for poly in polygons:
        try:
            mesh = trimesh.creation.extrude_polygon(poly, height=height)
            meshes.append(mesh)
        except Exception as e:
            # Print only the first 5 extrusion errors to prevent terminal flood
            if len(exceptions_list) < 5:
                exceptions_list.append(e)
                logger.error(f"Extrusion failed: {e.__class__.__name__}: {e}")
            continue
            
    return meshes


def process_city_model(vector_path: str, ndsm_path: str, output_glb: str):
    start_time = time.time()
    
    # 1. Load Data
    try:
        print("Loading vector data...")
        gdf = gpd.read_file(vector_path)
        src = rasterio.open(ndsm_path)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # 2. Handle CRS Alignment
    if gdf.crs != src.crs:
        print(f"Reprojecting GeoPackage from {gdf.crs} to match nDSM ({src.crs})...")
        gdf = gdf.to_crs(src.crs)

    # 3. Setup Coordinate Translation
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

    # 4. Tracking and Stats Setup
    stats = {
        'total': len(gdf),
        'valid': 0,
        'src_levels': 0,
        'src_ndsm': 0,
        'src_attr': 0,
        'src_default': 0,
        'skipped_invalid_geom': 0,
        'skipped_extrusion_fail': 0,
        'successful_meshes': 0
    }
    
    applied_heights = []
    city_meshes = []
    extrusion_exceptions = []

    # 5. Core Processing Loop
    with src:
        print("Processing building geometries...")
        for idx, row in tqdm(gdf.iterrows(), total=stats['total'], desc="Meshing City"):
            
            # Validate geometry
            valid_geom = validate_and_clean_geometry(row['geometry'])
            if not valid_geom:
                stats['skipped_invalid_geom'] += 1
                continue
                
            # Temporarily overwrite with clean 2D geometry for raster extraction
            row['geometry'] = valid_geom 

            # Extract Height based on priorities
            height, source = get_building_height(row, src)
            
            if source == 'levels':
                stats['src_levels'] += 1
            elif source == 'ndsm':
                stats['src_ndsm'] += 1
            elif source == 'attribute':
                stats['src_attr'] += 1
            else:
                stats['src_default'] += 1

            # Shift to Local Coordinates (Origin 0,0)
            local_geom = translate(valid_geom, xoff=-center_x, yoff=-center_y)

            # Generate LOD1 Mesh
            building_meshes = extrude_building(local_geom, height, extrusion_exceptions)
            
            if not building_meshes:
                stats['skipped_extrusion_fail'] += 1
                continue
                
            city_meshes.extend(building_meshes)
            applied_heights.append(height)
            stats['valid'] += 1
            stats['successful_meshes'] += len(building_meshes)

    duration = time.time() - start_time

    # 6. Compute Aggregates
    avg_h = np.mean(applied_heights) if applied_heights else 0.0
    min_h = np.min(applied_heights) if applied_heights else 0.0
    max_h = np.max(applied_heights) if applied_heights else 0.0

    # 7. Print Processing Summary
    print("\n==============================")
    print("PROCESSING SUMMARY")
    print("==============================")
    print(f"Total buildings                  : {stats['total']}")
    print(f"Buildings using building:levels  : {stats['src_levels']}")
    print(f"Buildings using nDSM             : {stats['src_ndsm']}")
    print(f"Buildings using attribute height : {stats['src_attr']}")
    print(f"Buildings using default height   : {stats['src_default']}")
    print(f"Average height                   : {avg_h:.2f} m")
    print(f"Minimum height                   : {min_h:.2f} m")
    print(f"Maximum height                   : {max_h:.2f} m")
    print(f"Invalid geometries skipped       : {stats['skipped_invalid_geom']}")
    print(f"Extrusion failures               : {stats['skipped_extrusion_fail']}")
    print(f"Successful meshes                : {stats['successful_meshes']}")
    print(f"Processing time                  : {duration:.2f} seconds")
    print("==============================\n")

    # 8. Export Mesh
    if not city_meshes:
        print("ERROR: No valid buildings were generated. Aborting export.")
        return

    print("Concatenating into a single city mesh...")
    final_mesh = trimesh.util.concatenate(city_meshes)
    
    print(f"Exporting to {output_glb}...")
    final_mesh.export(output_glb)


if __name__ == "__main__":
    BUILDING_FILE = "delhi_buildings.gpkg"
    NDSM_FILE = "ndsm.tif"
    OUTPUT_GLB = "city.glb"
    
    process_city_model(BUILDING_FILE, NDSM_FILE, OUTPUT_GLB)