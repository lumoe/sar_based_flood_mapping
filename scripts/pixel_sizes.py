from pprint import pprint
import os
import rasterio

from typing import List

# Get all tif files from 'data' folder recursively


def tif_files(path: str) -> List[str]:
    tif_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    return tif_files

# Get pixel size of all tif files


def pixel_sizes(tif_files: List[str]) -> List[float]:
    pixel_sizes = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            pixel_sizes.append(src.res)
    return pixel_sizes


if __name__ == '__main__':
    files = tif_files('data')
    sizes = pixel_sizes(files)
    for pixel_size in zip(files, sizes):
        if pixel_size[1] != (20, 20):
            print(pixel_size)
