import os
import glob

import rasterio

from typing import List, Tuple


def pixel_sizes(tif_files: List[str]) -> List[Tuple]:
    pixel_sizes = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            pixel_sizes.append((tif_file, src.res))
    return pixel_sizes


test_folder = os.path.join("data", "test")


def all_test_tiffs(folder: str):
    return glob.glob(folder + "/**/*.tif", recursive=True)


if __name__ == "__main__":
    for tif_file, pixel_size in pixel_sizes(all_test_tiffs(test_folder)):
        if pixel_size != (20, 20):
            print(tif_file)
