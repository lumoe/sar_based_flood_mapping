from pprint import pprint
import os
import rasterio

from typing import List

from config import DATA_DICT

# Get all tif files from 'data' folder recursively


def tif_files(path: str) -> List[str]:
    tif_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(root, file))
    return tif_files


# Get pixel size of all tif files


def pixel_sizes(tif_files: List[str]) -> List[float]:
    pixel_sizes = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            pixel_sizes.append(src.res)
    return pixel_sizes


if __name__ == "__main__":
    # files = tif_files('data')
    # sizes = pixel_sizes(files)
    # for pixel_size in zip(files, sizes):
    #     if pixel_size[1] != (20, 20):
    #         print(pixel_size)
    for _type in DATA_DICT["train"].keys():
        for location in DATA_DICT["train"][_type].keys():
            # print(_type, location)
            # pprint(DATA_DICT["train"][_type][location]["reference"])
            # pprint(DATA_DICT["train"][_type][location]["images"])
            # print()
            reference_files = tif_files(
                DATA_DICT["train"][_type][location]["reference"]
            )
            image_files = tif_files(DATA_DICT["train"][_type][location]["images"])
            reference_sizes = pixel_sizes(reference_files)
            image_sizes = pixel_sizes(image_files)
            for sizes in reference_sizes + image_sizes:
                print(sizes)
