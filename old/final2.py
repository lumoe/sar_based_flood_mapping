import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window, transform
from rasterio.plot import show
import tempfile
import os
from shapely.geometry import box
import geopandas as gpd

from tqdm import tqdm


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


def read_geotiff(file_path):
    with rasterio.open(file_path) as src:
        return src


def find_overlap_extent(src1, src2):
    extent1 = np.array(src1.bounds)
    extent2 = np.array(src2.bounds)

    overlap_extent = (max(extent1[0], extent2[0]), max(extent1[1], extent2[1]),
                      min(extent1[2], extent2[2]), min(extent1[3], extent2[3]))
    return overlap_extent


def create_chips(src1, src2, overlap_extent, chip_size=200, stride=100):
    x_min, y_min, x_max, y_max = overlap_extent
    chips = []

    row_start = y_min
    row_end = y_max - chip_size
    col_start = x_min
    col_end = x_max - chip_size

    for row in np.arange(row_start, row_end, stride):
        for col in np.arange(col_start, col_end, stride):
            window = rasterio.windows.from_bounds(
                col, row, col + chip_size, row + chip_size, src1.transform)
            chip1 = src1.read(window=window)
            chip2 = src2.read(window=window)
            chips.append((chip1, chip2))

            # if chip1.shape == chip2.shape:
            #     chips.append((chip1, chip2))
    return chips


target_train = 'Texas'

x_paths, y_paths = paths_train_reference_images(type='flood')
x_paths = [x for x in x_paths if target_train in x]
y_paths = [y for y in y_paths if target_train in y]

# Read the reference and Sentinel-1 images
ref_image_path = x_paths[0]
sentinel_image_path = y_paths[0]

with rasterio.open(ref_image_path) as src1:
    with rasterio.open(sentinel_image_path) as src2:

        # src1 = read_geotiff(ref_image_path)
        # src2 = read_geotiff(sentinel_image_path)

        # Find the overlapping extent
        overlap_extent = find_overlap_extent(src1, src2)

        # Create chips
        chips = create_chips(src1, src2, overlap_extent)

        # Close the rasterio objects to free resources
        # src1.close()
        # src2.close()

# print(chips)
for chip in chips:
    visualise_overlap(chip[0], chip[1])
