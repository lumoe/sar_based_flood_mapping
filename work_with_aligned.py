
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from rasterio.windows import Window, transform
from rasterio.plot import show
import tempfile
import os
from shapely.geometry import box
import geopandas as gpd

from rasterio.transform import from_bounds

# from final2 import visualise_overlap

import matplotlib.pyplot as plt

from rasterio.warp import calculate_default_transform, reproject, Resampling


from utils import paths_train_reference_images


def visualise_overlap(reference_data, sentinel_data, chip_name=""):
    reference_data = reference_data[0]
    sentinel_data = sentinel_data[0]

    # Replace the no data values with 0
    reference_data[reference_data == -9999] = 0
    sentinel_data[sentinel_data == -9999] = 0

    # Visualize the reference and Sentinel images side by side and on top of each other with 0.5 opacity
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(reference_data, cmap='gray')
    ax1.set_title("Reference Image")

    ax2.imshow(sentinel_data)
    ax2.set_title("Sentinel Image")

    ax3.imshow(reference_data, cmap='gray', alpha=0.5)
    ax3.imshow(sentinel_data, alpha=0.5)
    ax3.set_title("Overlay")

    for ax in [ax1, ax2, ax3]:
        ax.axis('off')

    plt.suptitle(chip_name)
    plt.show()


def create_chips(source1, source2, chip_size=1000, stride=1000):
    print("Creating chips...")
    # Align the two images to have the same extent and resolution
    dst_crs = source1.crs
    res = source1.transform.a
    width = max(source1.width, source2.width)
    height = max(source1.height, source2.height)

    aligned_data1 = np.empty(
        (source1.count, height, width), dtype=source1.meta['dtype'])
    aligned_data2 = np.empty(
        (source2.count, height, width), dtype=source2.meta['dtype'])

    for i in range(source1.count):
        reproject(
            source1.read(i + 1),
            aligned_data1[i],
            src_transform=source1.transform,
            src_crs=source1.crs,
            dst_transform=rasterio.transform.from_bounds(
                *source1.bounds, width, height),
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    for i in range(source2.count):
        reproject(
            source2.read(i + 1),
            aligned_data2[i],
            src_transform=source2.transform,
            src_crs=source2.crs,
            dst_transform=rasterio.transform.from_bounds(
                *source2.bounds, width, height),
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    # Create chips
    chips = []
    for y in range(0, height - chip_size, stride):
        for x in range(0, width - chip_size, stride):
            chip1 = aligned_data1[:, y:y + chip_size, x:x + chip_size]
            chip2 = aligned_data2[:, y:y + chip_size, x:x + chip_size]
            chips.append((chip1, chip2))

    return chips


target_train = 'Myanmar'

x_paths, y_paths = paths_train_reference_images(type='flood')
x_paths = [x for x in x_paths if target_train in x]
y_paths = [y for y in y_paths if target_train in y]


for chip in create_chips(rasterio.open(y_paths[0]), rasterio.open(x_paths[0])):
    visualise_overlap(chip[0], chip[1])
    # break
