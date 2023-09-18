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

from utils import get_train_image_pairs

from pprint import pprint


def read_with_bounds(x_img_src: DatasetReader, y_img_src: DatasetReader) -> np.ndarray:
    window = from_bounds(*x_img_src.bounds, transform=y_img_src.transform)
    y_img_boundless = y_img_src.read(window=window, boundless=True)

    # Replace nodata with 0
    y_img_boundless[y_img_boundless == y_img.nodata] = 0

    return y_img_boundless


def visualise_overlap(reference_data, sentinel_data, chip_name=""):
    # Visualize the reference and Sentinel images side by side and on top of each other with 0.5 opacity
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(reference_data, cmap="gray")
    ax1.set_title("Reference Image")

    ax2.imshow(sentinel_data)
    ax2.set_title("Sentinel Image")

    ax3.imshow(reference_data, cmap="gray", alpha=0.5)
    ax3.imshow(sentinel_data, alpha=0.5)
    ax3.set_title("Overlay")

    for ax in [ax1, ax2, ax3]:
        ax.axis("off")

    plt.suptitle(chip_name)
    plt.show()


from attrs import define
from typing import Literal

ImageReferenceType = Literal["image", "reference"]
WaterConditionType = Literal["water", "flood"]


@define
class Chip:
    image: np.ndarray
    x: int
    y: int
    location: str
    water_condition: WaterConditionType
    image_or_reference: ImageReferenceType
    path_prefix = os.path.join("data", "tmp")

    def not_empty(self) -> bool:
        return np.count_nonzero(self.image) > 0

    @property
    def filename(self) -> str:
        return f"{self.location}_{self.x}_{self.y}.tif"

    @property
    def image_normalized(self) -> np.ndarray:
        return (self.image - self.image.min()) / (self.image.max() - self.image.min())

    def save(self):
        if self.not_empty():
            with rasterio.open(
                os.path.join(
                    self.path_prefix,
                    self.water_condition,
                    self.image_or_reference,
                    self.filename,
                ),
                "w",
                width=self.image.shape[0],
                height=self.image.shape[1],
                count=1,
                dtype=self.image.dtype,
            ) as writer:
                writer.write(self.image, 1)


def chip_is_ok(chip: np.ndarray) -> bool:
    """
    Check if the chip is ok to be used for training
    chip: np.ndarray
    return: bool
    """
    # Check if chip is empty
    if np.count_nonzero(chip) == 0:
        return False

    # Check if the sum of the borders is 0
    if (
        np.sum(chip[0, :]) == 0
        or np.sum(chip[-1, :]) == 0
        or np.sum(chip[:, 0]) == 0
        or np.sum(chip[:, -1]) == 0
    ):
        return False

    return True


def create_chips(
    x: np.ndarray,
    y: np.ndarray,
    location: str,
    water_condition: WaterConditionType,
    size=256,
    stride=128,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    x_chips = []
    y_chips = []

    assert x.shape == y.shape

    for x_start in range(0, x.shape[0], stride):
        for y_start in range(0, x.shape[1], stride):
            x_chip = x[x_start : x_start + size, y_start : y_start + size]
            y_chip = y[x_start : x_start + size, y_start : y_start + size]
            if not chip_is_ok(x_chip):
                print(f"Skipping chip {x_start}_{y_start}")
                continue
            # Check that the chip is not empty
            x_chips.append(
                Chip(
                    x_chip,
                    x_start,
                    y_start,
                    location,
                    water_condition,
                    "image",
                )
            )
            y_chips.append(
                Chip(
                    y_chip,
                    x_start,
                    y_start,
                    location,
                    water_condition,
                    "reference",
                )
            )

    return x_chips, y_chips


def visualise_chips(chips: Tuple[List[Chip], List[Chip]]):
    x, y = chips
    for x, y in zip(x, y):
        if x.not_empty() and y.not_empty():
            assert x.image.shape == y.image.shape
            assert x.image.shape == (256, 256)
            assert y.image.shape == (256, 256)

            visualise_overlap(x.image, y.image, chip_name="Chip")


from tifffile import imwrite


def save_chips(chips: Tuple[List[Chip]]):
    i = 0
    x, y = chips
    for x, y in zip(x, y):
        if x.not_empty() and y.not_empty():
            # assert x.image.shape == y.image.shape
            # assert x.image.shape == (256, 256)
            # assert y.image.shape == (256, 256)

            if y.image.shape != (256, 256):
                # print(f"Skipping chip {y}")
                continue

            if x.image.shape != (256, 256):
                # print(f"Skipping chip {x}")
                continue

            # pprint(x.image_normalized)
            # pprint(y.image_normalized)

            x.save()
            y.save()

            i += 1

            # visualise_overlap(x.image, y.image, chip_name="Chip")


def plot_single_image(img: np.ndarray):
    # Matplotlib plot image
    # plt.imshow(img)
    show(img)


# def get_file_name()


image_type = ["water", "flood"]

image_type_paths = {
    "water": ["E033N009T3", "E042N012T3", "E051N015T3", "E051N027T3", "E060N021T3"],
    "flood": ["Texas2017", "Myanmar2019", "Greece2018"],
}

if __name__ == "__main__":
    for image_type in image_type_paths.keys():
        for image_location in image_type_paths[image_type]:
            print(image_type, image_location)
            train_image_paths = get_train_image_pairs()
            image_uri = train_image_paths[image_type][image_location]["images"]
            label_uri = train_image_paths[image_type][image_location]["reference"]

            pprint(image_uri)
            pprint(label_uri)

            with rasterio.open(image_uri[0], "r") as x_img, rasterio.open(
                label_uri[0], "r"
            ) as y_img:
                y_data = read_with_bounds(x_img, y_img)[0]

                x_data = x_img.read()[0]
                x_data[x_data == x_img.nodata] = 0

                chips = create_chips(
                    x_data,
                    y_data,
                    location=image_location,
                    water_condition=image_type,
                )

                assert len(chips[0]) == len(chips[0])

                # visualise_chips(chips)
                save_chips(chips)
