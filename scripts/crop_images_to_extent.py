from rastervision.core.data import RasterioSource
from utils import paths_train_reference_images, apply_mask_to_image_and_reference, mask_and_crop_images

import rasterio
import rasterio.features
import rasterio.warp
import rasterio.windows

from shapely.geometry import Polygon

from matplotlib import pyplot as plt

import numpy as np
import os

x_paths, y_paths = paths_train_reference_images(type='water')


for x, y in zip(x_paths, y_paths):

    with rasterio.open(x) as image_src, rasterio.open(y) as ref_src:
        # image = image_src.read(1)
        # image_meta = image_src.meta
        # image_bounds = image_src.bounds

        # ref_data = ref_src.read(1)
        # ref_meta = ref_src.meta
        # ref_bounds = ref_src.bounds

        # Replace all -999 values with 0
        # image[image == -999] = 0
        # ref_data[ref_data == -999] = 0

        # image_data_masked, ref_data_masked = apply_mask_to_image_and_reference(
        #     image_src, ref_src)
        image_data_masked, ref_data_masked = mask_and_crop_images(
            image_src, ref_src)

        # # Find the extent of image where there is data
        # image_data = np.ma.masked_equal(image, image_src.nodata)
        # # Remove all rows and columns that are all masked
        # image_data_masked = image_data[~image_data.mask.all(axis=1)]

        # # Remove columns that are all masked
        # image_data_masked = image_data_masked[:,
        #                                       ~image_data.mask.all(axis=0)]

        # # Remove the rows indices that are masked from the reference as well
        # ref_data = ref_data[~image_data.mask.all(axis=1)]
        # # Remove columns that are all masked
        # ref_data = ref_data[:, ~image_data.mask.all(axis=0)]

        # Plot image and ref_data side by side as images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plt.title(x)
        ax1.imshow(image_data_masked[0], cmap='gray')
        ax1.set_title('Image')
        ax2.imshow(ref_data_masked[0], cmap='gray')
        ax2.set_title('Reference')

        plt.show()

        cropped_image_path = os.path.join('data', 'cropped_test.tif')
        cropped_ref_path = os.path.join('data', 'cropped_test_ref.tif')

        with rasterio.open(cropped_image_path, 'w', **image_src.meta) as dst:
            dst.write(image_data_masked)

        with rasterio.open(cropped_ref_path, 'w', **ref_src.meta) as dst:
            dst.write(ref_data_masked)

    break
