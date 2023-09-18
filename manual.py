<<<<<<< HEAD

=======
>>>>>>> master
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

from rasterio import DatasetReader

from rasterio.windows import Window, transform
from rasterio.plot import show
import tempfile
import os
from shapely.geometry import box
import geopandas as gpd

from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from utils import paths_train_reference_images, paths_train_flood_images

from typing import List, Tuple

<<<<<<< HEAD
=======
from pprint import pprint

>>>>>>> master

def read_with_bounds(x_img_src: DatasetReader, y_img_src: DatasetReader) -> np.ndarray:
    window = from_bounds(*x_img_src.bounds, transform=y_img_src.transform)
    y_img_boundless = y_img_src.read(window=window, boundless=True)

    # Replace nodata with 0
    y_img_boundless[y_img_boundless == y_img.nodata] = 0

    return y_img_boundless


def visualise_overlap(reference_data, sentinel_data, chip_name=""):
    # Visualize the reference and Sentinel images side by side and on top of each other with 0.5 opacity
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

<<<<<<< HEAD
    ax1.imshow(reference_data, cmap='gray')
=======
    ax1.imshow(reference_data, cmap="gray")
>>>>>>> master
    ax1.set_title("Reference Image")

    ax2.imshow(sentinel_data)
    ax2.set_title("Sentinel Image")

<<<<<<< HEAD
    ax3.imshow(reference_data, cmap='gray', alpha=0.5)
=======
    ax3.imshow(reference_data, cmap="gray", alpha=0.5)
>>>>>>> master
    ax3.imshow(sentinel_data, alpha=0.5)
    ax3.set_title("Overlay")

    for ax in [ax1, ax2, ax3]:
<<<<<<< HEAD
        ax.axis('off')
=======
        ax.axis("off")
>>>>>>> master

    plt.suptitle(chip_name)
    plt.show()


<<<<<<< HEAD
def create_chips(x: np.ndarray, y: np.ndarray, size=200, stride=100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
=======
def create_chips(
    x: np.ndarray, y: np.ndarray, size=200, stride=100
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
>>>>>>> master
    x_chips = []
    y_chips = []

    assert x.shape == y.shape

    for x_start in range(0, x.shape[0], stride):
        for y_start in range(0, x.shape[1], stride):
<<<<<<< HEAD
            x_chips.append(
                x[x_start:x_start + size, y_start:y_start + size])
            y_chips.append(
                y[x_start:x_start + size, y_start:y_start + size])
=======
            x_chips.append(x[x_start : x_start + size, y_start : y_start + size])
            y_chips.append(y[x_start : x_start + size, y_start : y_start + size])
>>>>>>> master

    return x_chips, y_chips


def check_chip_not_empty(chip: np.ndarray) -> bool:
    return np.count_nonzero(chip) > 0


def visualise_chips(chips: Tuple[List[np.ndarray], List[np.ndarray]]):
    x, y = chips
    for x, y in zip(x, y):
        if check_chip_not_empty(x) and check_chip_not_empty(y):
<<<<<<< HEAD

=======
>>>>>>> master
            assert x.shape == y.shape
            assert x.shape == (200, 200)
            assert y.shape == (200, 200)

            visualise_overlap(x, y, chip_name="Chip")


def plot_single_image(img: np.ndarray):
    # Matplotlib plot image
    # plt.imshow(img)
    show(img)


<<<<<<< HEAD
x_paths, y_paths = paths_train_flood_images('Greece2018')

with rasterio.open(x_paths[0], 'r') as x_img, rasterio.open(y_paths[0], 'r') as y_img:

=======
x_paths, y_paths = paths_train_flood_images("Greece2018")

pprint(x_paths)
pprint(y_paths)


with rasterio.open(x_paths[0], "r") as x_img, rasterio.open(y_paths[0], "r") as y_img:
>>>>>>> master
    y_data = read_with_bounds(x_img, y_img)[0]

    x_data = x_img.read()[0]
    x_data[x_data == x_img.nodata] = 0

    chips = create_chips(x_data, y_data)

    assert len(chips[0]) == len(chips[0])

    visualise_chips(chips)
