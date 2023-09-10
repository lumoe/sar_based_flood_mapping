from rasterio.enums import Resampling
from rasterio import merge, warp
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window, from_bounds
from rasterio.plot import show
import tempfile
import os

from itertools import product


from typing import Literal


from rasterio.warp import calculate_default_transform, reproject, Resampling


from utils import paths_train_reference_images

x_paths, y_paths = paths_train_reference_images(type='flood')

target_train = 'Texas'
target_path = os.path.join('data', 'train', 'flood', target_train)

os.makedirs(target_path, exist_ok=True)

# go through x_paths and y_paths and remove the ones that conatin "Myanmar2019" anywhere in the string
x_paths = [x for x in x_paths if target_train in x]
y_paths = [y for y in y_paths if target_train in y]

# with rasterio.open(x_paths[0]) as src1, rasterio.open(y_paths[0]) as src2:

print(x_paths[0])
print(y_paths[0])


def read_and_reproject(geotiff_path, dst_crs):
    with rasterio.open(geotiff_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        dst_array = np.empty((src.count, height, width), dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, range(1, src.count + 1)),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
    return dst_array, kwargs


def create_chips(raster_array, raster_meta, window_size, output_dir, image_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open('temp.tif', 'w', **raster_meta) as temp:
        temp.write(raster_array)

    with rasterio.open('temp.tif') as src:
        ncols, nrows = src.meta['width'], src.meta['height']
        offsets = product(range(0, ncols, window_size),
                          range(0, nrows, window_size))
        for idx, (col_off, row_off) in enumerate(offsets):
            window = Window(
                col_off=col_off, row_off=row_off, width=window_size, height=window_size)
            transform = src.window_transform(window)
            chip = src.read(window=window)
            chip_meta = src.meta.copy()
            chip_meta.update({
                'transform': transform,
                'width': window.width,
                'height': window.height
            })

            with rasterio.open(os.path.join(output_dir, f'{image_type}_{idx}.tif'), 'w', **chip_meta) as dst:
                dst.write(chip)

            break

    os.remove('temp.tif')


sar_geotiff_path = x_paths[0]
gt_geotiff_path = y_paths[0]
window_size = 256

with rasterio.open(sar_geotiff_path) as src:
    dst_crs = src.crs

# Read and reproject SAR data
sar_data, sar_meta = read_and_reproject(sar_geotiff_path, dst_crs=dst_crs)

# Read and reproject ground truth data
gt_data, gt_meta = read_and_reproject(gt_geotiff_path, dst_crs=dst_crs)

# Create chips for SAR data
create_chips(sar_data, gt_meta, window_size, target_path, 'sar')

# Create chips for ground truth data
create_chips(gt_data, gt_meta, window_size, target_path, 'ref')
