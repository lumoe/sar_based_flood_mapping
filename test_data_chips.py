import os
import glob

import rasterio
from rasterio import DatasetReader
from rasterio.windows import from_bounds

from typing import List, Tuple


def pixel_sizes(tif_files: List[str]) -> List[Tuple]:
    pixel_sizes = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            pixel_sizes.append((tif_file, src.res))
    return pixel_sizes


TEST_FOLDER = os.path.join("data", "test")

from attrs import define
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

LocationType = Literal["Albania_2021", "Germany_2021", "Madagascar_2022"]
ImageType = Literal["masks", "tuw_results", "reference", "input"]


def read_with_bounds(x_img_src: DatasetReader, y_img_src: DatasetReader) -> np.ndarray:
    window = from_bounds(*x_img_src.bounds, transform=y_img_src.transform)
    y_img_boundless = y_img_src.read(window=window, boundless=True)

    # Replace nodata with 0
    # y_img_boundless[y_img_boundless == y_img_boundless.nodata] = 0

    return y_img_boundless


@define
class TestChip:
    image: np.ndarray
    x: int
    y: int
    location: LocationType
    image_type: ImageType
    path_prefix = os.path.join("data", "tmp", "test")

    @property
    def filename(self) -> str:
        return os.path.join(self.image_type, f"{self.location}_{self.x}_{self.y}.tif")

    def save(self):
        with rasterio.open(
            os.path.join(self.path_prefix, self.filename),
            "w",
            width=self.image.shape[0],
            height=self.image.shape[1],
            count=1,
            dtype=self.image.dtype,
        ) as writer:
            writer.write(self.image, 1)


def visualize_chip_overlaps(chips: List[TestChip]):
    # Subplots with amount of chips
    fig, axs = plt.subplots(1, len(chips), figsize=(20, 20))
    for i, chip in enumerate(chips):
        axs[i].imshow(chip.image)
        axs[i].set_title(chip.filename)

    for ax in axs:
        ax.axis("off")

    plt.show()


def create_chips(
    tuw_results: np.ndarray,
    input: np.ndarray,
    reference: np.ndarray,
    masks: np.ndarray,
    location: LocationType,
    size=256,
    stride=128,
) -> Tuple[List[TestChip], List[TestChip], List[TestChip], List[TestChip]]:
    tuw_results_chips: List[TestChip] = []
    input_chips: List[TestChip] = []
    reference_chips: List[TestChip] = []
    masks_chips: List[TestChip] = []

    assert tuw_results.shape == input.shape == reference.shape == masks.shape

    for x_start in range(0, tuw_results.shape[0], stride):
        for y_start in range(0, tuw_results.shape[1], stride):
            tuw_result_chip = tuw_results[x_start : x_start + size, y_start : y_start + size]  # fmt: skip
            input_chip = input[x_start : x_start + size, y_start : y_start + size]  # fmt: skip
            reference_chip = reference[x_start : x_start + size, y_start : y_start + size]  # fmt: skip
            masks_chip = masks[x_start : x_start + size, y_start : y_start + size]  # fmt: skip
            # if not chip_is_ok(tuw_result_chip):
            #     print(f"Skipping chip {x_start}_{y_start}")
            #     continue
            # Check that the chip is not empty
            tuw_results_chips.append(
                TestChip(
                    tuw_result_chip,
                    x_start,
                    y_start,
                    location,
                    "tuw_results",
                )
            )
            input_chips.append(
                TestChip(
                    input_chip,
                    x_start,
                    y_start,
                    location,
                    "input",
                )
            )
            reference_chips.append(
                TestChip(
                    reference_chip,
                    x_start,
                    y_start,
                    location,
                    "reference",
                )
            )
            masks_chips.append(
                TestChip(
                    masks_chip,
                    x_start,
                    y_start,
                    location,
                    "masks",
                )
            )

    return tuw_results_chips, input_chips, reference_chips, masks_chips


def create_chips_from_file(
    tuw_results_file: str,
    input_file: str,
    reference_file: str,
    masks_file: str,
    location: LocationType,
    size=256,
    stride=128,
) -> Tuple[List[TestChip], List[TestChip], List[TestChip], List[TestChip]]:
    with rasterio.open(tuw_results_file) as tuw_results_src:
        with rasterio.open(input_file) as input_src:
            with rasterio.open(reference_file) as reference_src:
                with rasterio.open(masks_file) as masks_src:
                    input = read_with_bounds(tuw_results_src, input_src)[0]
                    reference = read_with_bounds(tuw_results_src, reference_src)[0]
                    masks = read_with_bounds(tuw_results_src, masks_src)[0]

                    tuw_results = tuw_results_src.read()[0]

                    assert (
                        tuw_results_src.crs
                        == input_src.crs
                        == reference_src.crs
                        == masks_src.crs
                    )

                    # return None

                    chips = create_chips(
                        tuw_results, input, reference, masks, location, size, stride
                    )

                    return chips

    # return create_chips(tuw_results, input, reference, masks, location, size, stride)


def all_test_tiffs(folder: str):
    return glob.glob(folder + "/**/*.tif", recursive=True)


def save_chips(
    chips: Tuple[List[TestChip], List[TestChip], List[TestChip], List[TestChip]]
) -> None:
    tuw_results, input, reference, masks = chips

    for a, b, c, d in zip(tuw_results, input, reference, masks):
        if (
            a.image.shape != (256, 256)
            or b.image.shape != (256, 256)
            or c.image.shape != (256, 256)
            or d.image.shape != (256, 256)
        ):
            continue

        a.save()
        b.save()
        c.save()
        d.save()


from pprint import pprint

if __name__ == "__main__":
    data_folders = filter(
        lambda x: os.path.isdir(x) and not "_issue" in x,
        [os.path.join("data", "test", folder) for folder in os.listdir("data/test")],
    )

    for folders in data_folders:
        folders = sorted(
            all_test_tiffs(folders),
            key=lambda x: x.split("/")[3],
        )

        print(folders)

        input_file = folders[0]
        masks_file = folders[1]
        reference_file = folders[2]
        tuw_results_file = folders[3]

        chips = create_chips_from_file(
            tuw_results_file,
            input_file,
            reference_file,
            masks_file,
            folders[0].split("/")[2],
        )

        # a, b, c, d = chips
        # for i, x in enumerate(a):

        save_chips(chips)

        # if i >= 50 and i < 60:
        #     visualize_chip_overlaps([a[i], b[i], c[i], d[i]])
        #     print(folders)

    # for tif_file, pixel_size in pixel_sizes(all_test_tiffs(TEST_FOLDER)):
    #     if pixel_size != (20, 20):
    #         print(tif_file)
