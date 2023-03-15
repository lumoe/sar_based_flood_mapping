import rasterio
import rasterio.features
import rasterio.warp
import rasterio.crs

import os
from typing import List

from config import config

from rasterio.io import DatasetReader

from utils import paths_train_reference_images, path_to_all_images
from rasterio.warp import calculate_default_transform, reproject


def get_crs_for_flood(path: str, x_paths: List[str]) -> rasterio.CRS:
    if "Greece" in path:
        search = "Greece"
    elif "Myanmar" in path:
        search = "Myanmar"
    elif "Texas" in path:
        search = "Texas"

    x_paths = [x_path for x_path in x_paths if search.lower()
               in x_path.lower()]

    with rasterio.open(x_paths[0]) as src:
        src: DatasetReader = src
        return src.crs

    # with rasterio.open(path) as src:
    #     src: DatasetReader = src
    #     return src.crs


# with rasterio.open("/home/lukas/uni/sar_based_flood_mapping/data/train/Sentinel-1/floods/Myanmar2019/EQUI7_AS020M/E045N021T3/SIG0_20190716T113944__VV_A070_E045N021T3_AS020M_V0M2R4_S1AIWGRDH_TUWIEN.tif") as dataset:
# get crs from dataset

# mask = dataset.dataset_mask()
# for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
#     geom = rasterio.warp.transform_geom(
#         dataset.crs, "EPSG:4326", geom, precision=6)
#     print(geom)

x_paths, y_paths = path_to_all_images(type='flood')

for path in y_paths:
    with rasterio.open(path) as src:
        src: DatasetReader = src
        print()
        # print(dataset.transform)
        print(path)
        print(src.crs)
        print(src.bounds)
        dst_crs = get_crs_for_flood(path, x_paths)
        print(dst_crs)

        # # Calculate the transformation parameters for the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        # Define the new file output parameters
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Create the output file and reproject the GeoTIFF to the new CRS
        with rasterio.open(path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest)
