import rasterio
import numpy as np

with rasterio.open("mcd19a2_Optical_Depth_055_2025.tif") as src:
    band = src.read(1)

print("dtype:", band.dtype)
print("min:", band.min())
print("max:", band.max())
print("first 20 unique values:", np.unique(band)[:20])