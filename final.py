import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window, transform
from rasterio.plot import show
import tempfile
import os
from shapely.geometry import box
import geopandas as gpd


from rasterio.warp import calculate_default_transform, reproject, Resampling


from utils import paths_train_reference_images

target_train = 'Greece'

x_paths, y_paths = paths_train_reference_images(type='flood')
x_paths = [x for x in x_paths if target_train in x]
y_paths = [y for y in y_paths if target_train in y]

print(x_paths, y_paths)

found_samples = 0

target_path = os.path.join('data', 'tmp')

os.makedirs(target_path, exist_ok=True)


ref_src = rasterio.open(y_paths[0])
left, right, bottom, top = ref_src.bounds
width, height = ref_src.width, ref_src.height
print()


def chip_geotiff(input_file, output_folder, chip_size=200, stride=100):
    with rasterio.open(input_file) as src:
        width, height = src.width, src.height
        meta = src.meta.copy()

        for x in range(0, width, stride):
            for y in range(0, height, stride):
                chip_name = f"{x}_{y}.tif"
                output_path = os.path.join(output_folder, chip_name)

                window = Window(x, y, chip_size, chip_size)
                chip_data = src.read(window=window)

                meta.update({
                    "driver": "GTiff",
                    "height": chip_data.shape[1],
                    "width": chip_data.shape[2],
                    "transform": transform(window, src.transform),
                    # "dtype": "uint16"
                })

                with rasterio.open(output_path, "w", **meta) as dest:
                    dest.write(chip_data)


output_path_gt = os.path.join(target_path, 'gt')
output_path_sentinel = os.path.join(target_path, 'sentinel')

os.makedirs(output_path_gt, exist_ok=True)
os.makedirs(output_path_sentinel, exist_ok=True)

ground_truth = y_paths[0]
sentinel_path = x_paths[0]
clipped_sentinel_path = os.path.join(target_path, 'clipped_sentinel_1.tif')

# Clip Sentinel-1 image to the extent of the ground truth image
with rasterio.open(y_paths[0]) as gt_src:
    with rasterio.open(x_paths[0]) as s1_src:
        out_meta = s1_src.meta.copy()
        gt_bounds = gt_src.bounds
        gt_box = box(*gt_bounds)
        gt_geom = gpd.GeoSeries(gt_box, crs=gt_src.crs)
        # gt_geom = gt_geom.to_crs(crs=s1_src.crs)
        clipped_s1, clipped_transform = rasterio.mask.mask(
            s1_src, gt_geom.geometry, crop=True)
        out_meta.update({"driver": "GTiff",
                         "height": clipped_s1.shape[1],
                         "width": clipped_s1.shape[2],
                         "transform": clipped_transform})
        with rasterio.open(clipped_sentinel_path, "w", **out_meta) as dest:
            dest.write(clipped_s1)

# Generate chips for ground truth and clipped Sentinel-1 images
chip_geotiff(ground_truth, output_path_gt)
chip_geotiff(clipped_sentinel_path, output_path_sentinel)
