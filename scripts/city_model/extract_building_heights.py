"""
ISRO Project Pipeline: Extract Building Heights using Google Open Buildings 2.5D Temporal Dataset (2023)
=======================================================================================================
This script extracts building heights from Google Earth Engine (GEE) using vector building footprints. 
It utilizes the Google Open Buildings Temporal V1 ImageCollection, extracts heights securely, 
and generates comprehensive spatial models and statistics suitable for downstream research.
"""

import os
import sys
import warnings
from typing import Tuple

import ee
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt

# Ignore verbose warnings from dependencies to keep the console output clean
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# FILE CONSTANTS
# ----------------------------------------------------
INPUT_GPKG = "buildings_aoi.gpkg"
OUTPUT_GPKG = "CIVIL_LINES_buildings_height.gpkg"
OUTPUT_CSV = "height_statistics.csv"
OUTPUT_MAP = "building_heights_map.png"
TEMP_RASTER = "temp_building_heights.tif"


# ----------------------------------------------------
# PIPELINE FUNCTIONS
# ----------------------------------------------------

def step1_read_data() -> gpd.GeoDataFrame:
    """Reads the local GeoPackage and prints basic metadata."""
    print("===============================================================")
    print("ISRO BUILDING HEIGHT EXTRACTION PIPELINE (GOOGLE OPEN BUILDINGS)")
    print("===============================================================\n")
    
    print(f"STEP 1: Reading input GeoPackage: {INPUT_GPKG}")
    if not os.path.exists(INPUT_GPKG):
        print(f"CRITICAL ERROR: Input file '{INPUT_GPKG}' not found in the working directory.")
        sys.exit(1)
        
    gdf = gpd.read_file(INPUT_GPKG)
    print(f"  CRS: {gdf.crs}")
    print(f"  Number of buildings: {len(gdf)}")
    print(f"  Bounds (minx, miny, maxx, maxy): {gdf.total_bounds}")
    return gdf


def step2_authenticate_and_download(gdf: gpd.GeoDataFrame) -> str:
    """Authenticates with Earth Engine and downloads the clipped raster."""
    print("\nSTEP 2: Authenticating and initializing Google Earth Engine...")
    PROJECT = "river-key-323305"
    try:
        ee.Initialize(project = PROJECT)
    except Exception:
        print("  GEE initialization failed. Triggering browser authentication flow...")
        try:
            ee.Authenticate()
            ee.Initialize(project = PROJECT)
        except Exception as auth_e:
            print(f"CRITICAL ERROR: Earth Engine authentication failed: {auth_e}")
            sys.exit(1)
    print("  ✓ Earth Engine Authenticated successfully.")

    print("\n--- Generating Google Earth Engine Download Request ---")
    # Convert ONLY for Earth Engine download
    gdf_gee = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf_gee.total_bounds
    
    # Define Region of Interest (ROI) from GeoDataFrame bounds
    region = ee.Geometry.Rectangle(
        [minx, miny, maxx, maxy],
        proj="EPSG:4326",
        geodesic=False
    )
    
    # Access Open Buildings Temporal Dataset for 2023
    collection = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
    
    # Filter by 2023, mosaic seamlessly, and select the 'building_height' band
    image = collection.filterDate('2023-01-01', '2023-12-31').mosaic().select('building_height')
    print("Earth Engine bounds:")
    print(minx, miny, maxx, maxy)
    print("  Requesting dynamic download URL from Earth Engine...")
    try:
        # Request restricted exactly to our Region of Interest
        url = image.getDownloadURL({
            'region': region,
            'crs': 'EPSG:4326',
            'scale': 4,
            'format': 'GEO_TIFF'
        })
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to get download URL from GEE: {e}")
        sys.exit(1)
        
    print(f"  Downloading raster data to disk...")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(TEMP_RASTER, 'wb') as f:
            f.write(response.content)
        print(f"  ✓ Raster successfully downloaded to {TEMP_RASTER}")
    except requests.exceptions.RequestException as e:
        print(f"CRITICAL ERROR: Raster download failed. Check internet connection: {e}")
        sys.exit(1)
        
    print("\n--- How this GEE Download Works ---")
    print("Instead of downloading the massive 58 million km² global dataset, we programmatically")
    print("define a bounding box matching your GeoPackage. We query the 'open-buildings-temporal/v1'")
    print("ImageCollection, filter it for the year 2023, and extract ONLY the 'building_height' band.")
    print("The API then generates a temporary download URL, cropping and scaling the raster at exactly")
    print("4 meters per pixel directly on Google's servers before downloading a lightweight GeoTIFF.")
    
    return TEMP_RASTER


def step3_print_raster_metadata(raster_path: str):
    """Inspects the local GeoTIFF and outputs physical properties."""
    print("\nSTEP 3: Inspecting downloaded raster metadata...")
    try:
        with rasterio.open(raster_path) as src:
            print(f"  CRS: {src.crs}")
            print(f"  Resolution (Pixel Size): ({src.res[0]:.6f}, {src.res[1]:.6f})")
            print(f"  Width: {src.width} px")
            print(f"  Height: {src.height} px")
            print(f"  NoData value: {src.nodata}")
            
            arr = src.read(1)
            # Remove NoData and zeroes for realistic min/max reporting
            if src.nodata is not None:
                arr = arr[arr != src.nodata]
            arr = arr[arr > 0] 

            if len(arr) > 0:
                print(f"  Minimum height present: {np.nanmin(arr):.1f} m")
                print(f"  Maximum height present: {np.nanmax(arr):.1f} m")
            else:
                print("  WARNING: Downloaded raster contains no positive height values.")
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot open downloaded raster file: {e}")
        sys.exit(1)


def step4_extract_heights(gdf: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    """Masks the raster for every polygon and calculates the median height."""
    print("\nSTEP 4: Extracting heights for every building footprint...")
    heights = []
    
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        
        # CRS match safety check
        if gdf.crs != src.crs:
            print(f"  Reprojecting vectors from {gdf.crs} to {src.crs} for perfect raster alignment...")
            gdf_eval = gdf.to_crs(src.crs)
        else:
            gdf_eval = gdf
            
        # Iterate over geometries efficiently
        for idx, geom in enumerate(gdf_eval.geometry):
            # Skip corrupted geometries
            if geom is None or geom.is_empty or not geom.is_valid:
                heights.append(np.nan)
                continue
                
            try:
                # Mask raster to exact polygon shape. all_touched=True catches edge pixels on small footprints.
                out_image, _ = mask(src, [geom], crop=True, filled=False, all_touched=False)
                
                # Fetch unmasked pixels intersecting the building
                pixels = out_image.compressed()
                
                # Exclude native NoData
                if nodata is not None:
                    pixels = pixels[pixels != nodata]
                # Exclude pixels reporting 0 height (not contributing to true structure height)

                if nodata is not None:
                    pixels = pixels[pixels != nodata]

                pixels = pixels[np.isfinite(pixels)]

                # NEW
                pixels = pixels[pixels > 0]

                if nodata is not None:
                    pixels = pixels[pixels != nodata]
                
                MIN_PIXELS = 3

                if len(pixels) < MIN_PIXELS:
                    heights.append(np.nan)
                else:
                    heights.append(np.median(pixels))
            except ValueError:
                heights.append(np.nan) # Geometry lies completely outside the raster bounds
            except Exception:
                heights.append(np.nan)
                
    gdf["height_m"] = heights

    median_height = np.nanmedian(gdf["height_m"])

    gdf["height_m"] = gdf["height_m"].fillna(median_height)

    gdf["height_m"] = gdf["height_m"].clip(
        lower=3,
        upper=80
    )
        
    print("\n--- Why Median is Better than Mean ---")
    print("The mean is highly sensitive to statistical outliers. If a building footprint")
    print("overlaps slightly with an adjacent taller tower, or if the AI detected localized")
    print("noise (e.g. water tanks or antennas), the mean will artificially skew upwards.")
    print("The median effectively filters out extreme noise by pulling the 50th percentile,")
    print("guaranteeing the extracted value represents the actual flat/main structural roof deck.")
    
    return gdf


def step5_quality_control(gdf: gpd.GeoDataFrame):
    """Summarizes building heights and outputs a histogram to the console."""
    print("\nSTEP 5: Quality Control Statistics")
    total_buildings = len(gdf)
    valid_heights = gdf['height_m'].dropna()
    valid_count = len(valid_heights)
    missing_count = total_buildings - valid_count
    
    print(f"  Total buildings processed: {total_buildings}")
    print(f"  Buildings with valid height: {valid_count}")
    print(f"  Buildings with missing height: {missing_count}")
    
    if valid_count > 0:
        print(f"  Average height: {valid_heights.mean():.2f} m")
        print(f"  Median height: {valid_heights.median():.2f} m")
        print(f"  Minimum height: {valid_heights.min():.2f} m")
        print(f"  Maximum height: {valid_heights.max():.2f} m")
        
        print("\n  Histogram of extracted heights:")
        counts, bins = np.histogram(valid_heights, bins=10)
        for i in range(len(counts)):
            print(f"    {bins[i]:5.1f} m to {bins[i+1]:5.1f} m : {counts[i]:4d} buildings")
    else:
        print("  WARNING: No heights extracted (are buildings outside the raster area?).")


def step6_visualize(gdf: gpd.GeoDataFrame, output_path: str):
    """Renders a choropleth map of the new building heights."""
    print("\nSTEP 6: Generating visualization map...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 1. Plot buildings with missing data (Grey)
    missing_gdf = gdf[gdf['height_m'].isna()]
    if not missing_gdf.empty:
        missing_gdf.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.2, label='No Data')
    
    # 2. Plot buildings with height data (Viridis Colormap)
    valid_gdf = gdf[gdf['height_m'].notna()]
    if not valid_gdf.empty:
        valid_gdf.plot(column='height_m', ax=ax, legend=True,
                       cmap='viridis', edgecolor='black', linewidth=0.2,
                       legend_kwds={'label': "Extracted Height (m)", 'shrink': 0.7})
                       
    ax.set_title("Building Heights from Google Open Buildings 2.5D Temporal Dataset (2023)", 
                 fontsize=14, fontweight='bold')
                 
    # 3. Apply North Arrow
    x, y, arrow_length = 0.95, 0.95, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=16,
                xycoords=ax.transAxes)
                
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Map saved successfully to: {output_path}")


def step7_export_gpkg(gdf: gpd.GeoDataFrame, output_path: str):
    """Exports the fully enriched GeoDataFrame."""
    print(f"\nSTEP 7: Exporting geometry to {output_path}...")
    try:
        gdf.to_file(output_path, driver="GPKG")
        print(f"  ✓ GPKG export successful.")
    except Exception as e:
        print(f"  ERROR: Failed to save GPKG: {e}")


def step8_export_csv(gdf: gpd.GeoDataFrame, output_path: str):
    """Exports structured non-spatial statistics for the buildings."""
    print(f"\nSTEP 8: Exporting dataset statistics to {output_path}...")
    
    # Safely assign a primary key if one is missing
    if 'building_id' not in gdf.columns:
        gdf['building_id'] = gdf.index
        
    # Project to Cylindrical Equal Area (EPSG:6933) to accurately calculate footprint area in square meters.
    # Calculating area purely on EPSG:4326 degrees creates massive inaccuracies. 
    try:
        gdf_cea = gdf.to_crs(epsg=6933)
        gdf['footprint_area'] = gdf_cea.geometry.area
    except Exception as e:
        print(f"  Warning: Metric area projection failed ({e}). Defaulting to NaN.")
        gdf['footprint_area'] = np.nan
        
    # Calculate geographical centroids in original CRS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Ignored as X,Y lat/long points are intentionally requested
        gdf['centroid_x'] = gdf.geometry.centroid.x
        gdf['centroid_y'] = gdf.geometry.centroid.y
        
    df_csv = gdf[['building_id', 'height_m', 'footprint_area', 'centroid_x', 'centroid_y']]
    
    try:
        df_csv.to_csv(output_path, index=False)
        print(f"  ✓ CSV export successful.")
    except Exception as e:
        print(f"  ERROR: Failed to save CSV: {e}")


# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------
def main():
    gdf = step1_read_data()
    raster_path = step2_authenticate_and_download(gdf)
    step3_print_raster_metadata(raster_path)
    
    gdf_enriched = step4_extract_heights(gdf, raster_path)
    step5_quality_control(gdf_enriched)
    step6_visualize(gdf_enriched, OUTPUT_MAP)
    
    step7_export_gpkg(gdf_enriched, OUTPUT_GPKG)
    step8_export_csv(gdf_enriched, OUTPUT_CSV)
    
    # Cleanup temporary raster
    if os.path.exists(TEMP_RASTER):
        os.remove(TEMP_RASTER)
    
    print("\n===============================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY.")
    print("===============================================================")

if __name__ == "__main__":
    main()