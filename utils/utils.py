import glob
import os

from config import config
from typing import Literal, Tuple, List

import numpy as np

from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio import mask, warp

import rasterio

AVAILABLE_FLOOD_EVENTS = ['Greece2018', 'Myanmar2019', 'Texas2017']


def get_crs_for_file(file_path: str) -> str:
    with rasterio.open(file_path) as src:
        return src.crs


def paths_train_flood_images(event: Literal['Greece2018', 'Myanmar2019', 'Texas2017']) -> Tuple[List[str], List[str]]:
    images_folder_filter = os.path.join(
        config.data.train.images_water if type == "water" else config.data.train.images_flood, '**/*.tif')
    reference_folder_filter = os.path.join(
        config.data.train.reference_water if type == "water" else config.data.train.reference_flood, '**/*.tif')
    paths_X, paths_Y = glob.glob(images_folder_filter, recursive=True), glob.glob(
        reference_folder_filter, recursive=True)

    # go through x_paths and y_paths and remove the ones that conatin "Myanmar2019" anywhere in the string
    paths_X = [x for x in paths_X if event in x]
    paths_Y = [y for y in paths_Y if event in y]

    return paths_X, paths_Y


def paths_train_reference_images(type: Literal["water", "flood"]) -> Tuple[List[str], List[str]]:
    images_folder_filter = os.path.join(
        config.data.train.images_water if type == "water" else config.data.train.images_flood, '**/*.tif')
    reference_folder_filter = os.path.join(
        config.data.train.reference_water if type == "water" else config.data.train.reference_flood, '**/*.tif')
    paths_X, paths_Y = glob.glob(images_folder_filter, recursive=True), glob.glob(
        reference_folder_filter, recursive=True)

    # Filter out paths that do do not have TILED in the name
    # paths_X = [path for path in paths_X if "TILED" not in path]
    # paths_Y = [path for path in paths_Y if "TILED" not in path]

    return paths_X, paths_Y


def path_to_all_images(type: Literal["water", "flood"]) -> Tuple[List[str], List[str]]:
    images_folder_filter = os.path.join(
        config.data.train.images_water if type == "water" else config.data.train.images_flood, '**/*.tif')
    reference_folder_filter = os.path.join(
        config.data.train.reference_water if type == "water" else config.data.train.reference_flood, '**/*.tif')
    paths_X, paths_Y = glob.glob(images_folder_filter, recursive=True), glob.glob(
        reference_folder_filter, recursive=True)

    return paths_X, paths_Y


def apply_mask_to_image_and_reference(image_src: DatasetReader, reference_src: DatasetReader) -> Tuple[np.ndarray, np.ndarray]:
    image_data = image_src.read(1)
    ref_data = reference_src.read(1)

    # Find the extent of original image and reference image where there is data
    image_mask = np.ma.masked_equal(image_data, image_src.nodata)
    ref_data_mask = np.ma.masked_equal(ref_data, reference_src.nodata)

    # Apply the image mask to both the image and the reference

    # Remove all rows and columns that are all masked
    image_data_masked = image_mask[~image_mask.mask.all(axis=1)]

    # Remove columns that are all masked
    image_data_masked = image_data_masked[:,
                                          ~image_mask.mask.all(axis=0)]

    # Remove the rows indices that are masked from the reference as well
    ref_data_masked = ref_data[~image_mask.mask.all(axis=1)]
    # Remove columns that are all masked
    ref_data_masked = ref_data_masked[:, ~image_mask.mask.all(axis=0)]

    # Apply the ref_data_mask to the image and the reference

    if ref_data_mask.shape[1] <= image_data_masked.shape[1]:
        # Remove all rows and columns that are all masked
        image_data_masked = image_data_masked[~ref_data_mask.mask.all(axis=1)]

    if ref_data_mask.shape[0] <= image_data_masked.shape[0]:
        # Remove columns that are all masked
        image_data_masked = image_data_masked[:,
                                              ~ref_data_mask.mask.all(axis=0)]

    if ref_data_mask.shape[1] <= ref_data_masked.shape[1]:
        # Remove the rows indices that are masked from the reference as well
        ref_data_masked = ref_data_masked[~ref_data_mask.mask.all(axis=1)]

    if ref_data_mask.shape[0] <= ref_data_masked.shape[0]:
        # Remove columns that are all masked
        ref_data_masked = ref_data_masked[:, ~ref_data_mask.mask.all(axis=0)]

    return image_data_masked, ref_data_masked


def mask_and_crop_images(src1: DatasetReader, src2: DatasetReader) -> Tuple[np.ndarray, np.ndarray]:
    # Get the nodata values as integers
    nodata1 = int(src1.nodata)
    nodata2 = int(src2.nodata)

    # Define the geometry as the entire image extent
    geometry = [{
        'type': 'Polygon',
        'coordinates': [(
            (src1.bounds.left, src1.bounds.bottom),
            (src1.bounds.right, src1.bounds.bottom),
            (src1.bounds.right, src1.bounds.top),
            (src1.bounds.left, src1.bounds.top),
            (src1.bounds.left, src1.bounds.bottom)
        )]
    }]

    # Reproject the geometry to match the CRS of the input images
    geometry = warp.transform_geom(src1.crs, src1.crs, geometry, precision=6)

    # Crop the two images based on the geometry and nodata values
    crop1, crop1_transform = mask.mask(
        src1, geometry, crop=True, all_touched=True, nodata=nodata1)
    crop2, crop2_transform = mask.mask(
        src2, geometry, crop=True, all_touched=True, nodata=nodata2)

    return crop1, crop2
