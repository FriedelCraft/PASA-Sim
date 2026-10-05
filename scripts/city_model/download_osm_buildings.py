import sys
import difflib
from typing import Optional, Tuple, Union
import fiona
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

# Enable KML support in Fiona (required for GeoPandas to read KML files)
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Configure OSMnx settings (Timeout = 300s, Enable Caching)
ox.settings.timeout = 300
ox.settings.use_cache = True


def load_kml_wards(file_path: str) -> gpd.GeoDataFrame:
    """Loads a KML file into a GeoDataFrame."""
    try:
        gdf = gpd.read_file(file_path, driver='KML')
        if gdf.empty:
            raise ValueError("The KML file was loaded but contains no data.")
        return gdf
    except Exception as e:
        print(f"❌ ERROR: Failed to read KML file '{file_path}'.\nDetails: {e}")
        sys.exit(1)


def get_ward_name_column(gdf):
    """
    Find the best column containing ward names.
    """

    preferred = [
        "WardName",
        "WARDNAME",
        "Ward_Name",
        "Name",
        "name",
        "WARD_NAME",
        "ward"
    ]

    for col in preferred:
        if col in gdf.columns:
            if gdf[col].notna().sum() > 0:
                return col

    # Fallback: first non-empty string column
    for col in gdf.columns:
        if gdf[col].dtype == object and gdf[col].notna().sum() > 0:
            return col

    raise ValueError("No suitable ward name column found.")


def select_ward(gdf: gpd.GeoDataFrame, name_col: str) -> Tuple[gpd.GeoDataFrame, str]:
    """Prompts the user to select a ward by number or name with robust error handling."""
    ward_names = gdf[name_col].astype(str).tolist()
    
    print("\n" + "="*40)
    print("AVAILABLE WARDS")
    print("="*40)
    for i, name in enumerate(ward_names, start=1):
        print(f"{i}. {name}")
    print("="*40)

    while True:
        user_input = input("\nEnter the ward NUMBER or NAME: ").strip()
        
        # 1. Check if input is a number
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(ward_names):
                selected_name = ward_names[idx]
                return gdf.iloc[[idx]].copy(), selected_name
            else:
                print(f"⚠️ Invalid number. Please enter a number between 1 and {len(ward_names)}.")
                continue
                
        # 2. Check if input is an exact name (case-insensitive)
        lower_names = [n.lower() for n in ward_names]
        if user_input.lower() in lower_names:
            idx = lower_names.index(user_input.lower())
            selected_name = ward_names[idx]
            return gdf.iloc[[idx]].copy(), selected_name
            
        # 3. Handle missing/typo inputs
        close_matches = difflib.get_close_matches(user_input.lower(), lower_names, n=3, cutoff=0.5)
        if close_matches:
            # Map back to original casing
            suggestions = [ward_names[lower_names.index(m)] for m in close_matches]
            print(f"⚠️ Ward '{user_input}' not found. Did you mean:")
            for s in suggestions:
                print(f"   - {s}")
        else:
            print(f"⚠️ Ward '{user_input}' not found. No similar matches detected. Try entering the number.")


def clean_geometry(geom: BaseGeometry) -> Optional[Union[Polygon, MultiPolygon]]:
    """Repairs invalid geometries and filters out non-polygonal artifacts."""
    if geom is None or geom.is_empty:
        return None

    if not geom.is_valid:
        geom = make_valid(geom)

    # Sometimes make_valid creates GeometryCollections containing Points/Lines. 
    # Extract only the polygonal parts.
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        if not polys:
            return None
        geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]

    # Buffer 0 fallback for stubborn self-intersections
    if not geom.is_valid:
        geom = geom.buffer(0)
        
    if geom.geom_type not in ['Polygon', 'MultiPolygon']:
        return None

    return geom


def sanitize_for_gpkg(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Converts OSM list/dict tags into strings so GeoPackage doesn't crash on export."""
    for col in gdf.columns:
        # Check if the column contains unhashable or nested types
        if gdf[col].apply(lambda x: isinstance(x, (list, dict))).any():
            gdf[col] = gdf[col].astype(str)
    return gdf


def main():
    KML_FILE = "delhi_wards.kml"
    
    # STEP 1: Read KML
    print("Initializing...")
    wards_gdf = load_kml_wards(KML_FILE)
    name_col = get_ward_name_column(wards_gdf)
    
    # STEP 2: Select Ward
    ward_gdf, ward_name = select_ward(wards_gdf, name_col)
    
    print(f"\nTargeting ward: {ward_name}")
    safe_ward_name = ward_name.replace(" ", "_").replace("/", "_")
    
    # STEP 3: Extract and repair ward polygon
    original_geom = ward_gdf.geometry.iloc[0]
    clean_ward_geom = clean_geometry(original_geom)
    
    if clean_ward_geom is None:
        print("❌ ERROR: Ward geometry is empty or unrecoverable. Exiting.")
        sys.exit(1)
        
    ward_gdf.geometry = [clean_ward_geom]
    
    # Calculate area (Project to UTM automatically via OSMnx to get meters, then convert to km^2)
    try:
        ward_proj = ox.project_gdf(ward_gdf)
        area_km2 = ward_proj.area.iloc[0] / 1_000_000
    except Exception:
        area_km2 = 0.0 # Fallback if projection fails

    # STEP 4: Download buildings
    print(f"Downloading OpenStreetMap buildings for {ward_name} (Timeout: 300s)...")
    tags = {"building": True}
    
    try:
        buildings_gdf = ox.features_from_polygon(clean_ward_geom, tags)
    except Exception as e:
        print(f"\n❌ ERROR: OSMnx download failed.")
        print(f"Possible reasons: Overpass server timeout, invalid polygon, or no internet.")
        print(f"Details: {e}")
        sys.exit(1)

    if buildings_gdf.empty:
        print(f"\n❌ ERROR: Download succeeded, but 0 buildings were found in {ward_name}.")
        sys.exit(1)

    # STEP 5: Filter, clean, and reset index
    print("Cleaning downloaded geometries...")
    # Keep only Polygons and MultiPolygons
    allowed_types = ['Polygon', 'MultiPolygon']
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(allowed_types)].copy()
    
    # Remove duplicates (OSMnx uses index as unique ID [element_type, osmid])
    buildings_gdf = buildings_gdf[~buildings_gdf.index.duplicated(keep='first')]
    
    # Reset index so 'osmid' and 'element_type' become standard columns
    buildings_gdf.reset_index(inplace=True)
    
    total_buildings = len(buildings_gdf)
    poly_count = (buildings_gdf.geometry.type == 'Polygon').sum()
    multipoly_count = (buildings_gdf.geometry.type == 'MultiPolygon').sum()

    # STEP 6 & 7: Export to GeoPackage
    buildings_out = f"{safe_ward_name}_buildings.gpkg"
    boundary_out = f"{safe_ward_name}_boundary.gpkg"
    
    print("Saving to GeoPackage...")
    try:
        # Sanitize list/dict tags before export
        buildings_gdf = sanitize_for_gpkg(buildings_gdf)
        
        buildings_gdf.to_file(buildings_out, driver="GPKG")
        ward_gdf.to_file(boundary_out, driver="GPKG")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save GeoPackages. Ensure the files are not open in another program.")
        print(f"Details: {e}")
        sys.exit(1)

    # STEP 8: Processing Summary
    print("\n" + "="*37)
    print("DOWNLOAD SUMMARY")
    print("="*37)
    print(f"Ward selected:\n{ward_name}\n")
    print(f"Boundary area:\n{area_km2:.2f} km²\n")
    print(f"Buildings downloaded:\n{total_buildings}\n")
    print(f"Polygons:\n{poly_count}\n")
    print(f"Multipolygons:\n{multipoly_count}\n")
    print("Output files:")
    print(f"- {buildings_out}")
    print(f"- {boundary_out}")
    print("="*37)


if __name__ == "__main__":
    main()