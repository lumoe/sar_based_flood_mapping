from rasterio.transform import Affine
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import tempfile


from utils import paths_train_reference_images

x_paths, y_paths = paths_train_reference_images(type='flood')

found_samples = 0

target_train = 'Texas'

target_path = os.path.join('data', 'train', 'flood', target_train)

os.makedirs(target_path, exist_ok=True)


def reproject_and_align(src_img, dst_img, dst_crs):
    transform, width, height = calculate_default_transform(
        src_img.crs, dst_crs, src_img.width, src_img.height, *src_img.bounds
    )

    kwargs = src_img.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(dst_img, 'w', **kwargs) as dst:
        for i in range(1, src_img.count + 1):
            reproject(
                source=rasterio.band(src_img, i),
                destination=rasterio.band(dst, i),
                src_transform=src_img.transform,
                src_crs=src_img.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )


sar_image_path = x_paths[0]
gt_image_path = y_paths[0]

sar_src = rasterio.open(sar_image_path)
gt_src = rasterio.open(gt_image_path)
# with rasterio.open(gt_image_path) as gt_src:
if sar_src.crs != gt_src.crs:
    reprojected_sar_path = '/tmp/reprojected_sar_image.tif'
    reproject_and_align(sar_src, reprojected_sar_path, gt_src.crs)
    sar_src.close()
    sar_src = rasterio.open(reprojected_sar_path)

    # Proceed with aligned images


def create_chips(image, chip_size=256, stride=256):
    chips = []
    height, width = image.shape
    for y in range(0, height - chip_size + 1, stride):
        for x in range(0, width - chip_size + 1, stride):
            chip = image[y:y + chip_size, x:x + chip_size]
            chips.append(chip)
    return chips


sar_band = sar_src.read(1)
gt_band = gt_src.read(1)

sar_chips = create_chips(sar_band)
gt_chips = create_chips(gt_band)


def save_chips(chips, src_meta, output_dir, prefix):
    output_dir = os.path.join(output_dir, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chip in enumerate(chips):
        chip_meta = src_meta.copy()
        chip_meta.update({
            'height': chip.shape[0],
            'width': chip.shape[1],
            'transform': src_meta['transform'] * Affine.translation(i * chip.shape[1], 0),
        })

        chip_path = os.path.join(output_dir, f"{i:03d}.tif")

        with rasterio.open(chip_path, 'w', **chip_meta) as chip_dst:
            chip_dst.write(chip, 1)

        # break


# save_chips(sar_chips, gt_src.meta, target_path, 'sar')
save_chips(gt_chips, gt_src.meta, target_path, 'gt')
