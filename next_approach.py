import rasterio
import geopandas as gpd
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import box

from utils import paths_train_reference_images
import numpy as np

x_paths, y_paths = paths_train_reference_images(type='flood')

found_samples = 0

# go through x_paths and y_paths and remove the ones that conatin "Myanmar2019" anywhere in the string
x_paths = [x for x in x_paths if 'Texas' in x]
y_paths = [y for y in y_paths if 'Texas' in y]


def get_reference_bounds_as_polygon(ref_image_path):
    with rasterio.open(ref_image_path) as src:
        bounds = src.bounds
        crs = src.crs
    return gpd.GeoDataFrame({'geometry': [box(*bounds)]}, crs=crs)


def crop_sar_to_reference(sar_image_path, reference_bounds):
    with rasterio.open(sar_image_path) as src:
        sar_data, sar_transform = mask(
            src, reference_bounds.geometry, crop=True)
    return sar_data, sar_transform


def log_scale_sar_data(sar_data, factor=10.0):
    return np.log10(sar_data * factor)


reference_image_path = y_paths[0]
with rasterio.open(reference_image_path) as src:
    reference_data = src.read(1)
    reference_transform = src.transform
    reference_crs = src.crs

sar_image_path = x_paths[0]
reference_bounds = get_reference_bounds_as_polygon(reference_image_path)
sar_data, sar_transform = crop_sar_to_reference(
    sar_image_path, reference_bounds)
sar_data_log = log_scale_sar_data(sar_data)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

show(reference_data, transform=reference_transform, ax=ax[0])
ax[0].set_title('Reference Image')

show(sar_data_log[0], transform=sar_transform, ax=ax[1], cmap='gray')
ax[1].set_title('Cropped SAR Image')

plt.show()
