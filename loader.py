from typing import List, Tuple, Literal

import matplotlib.pyplot as plt

import os

import rasterio

import numpy as np


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


def load_samples(path: str):
    # Read all files from path
    image_files = os.listdir(os.path.join(path, "image"))

    i = 0
    for image_file in image_files:
        print(image_file)
        reference_file = os.path.join(path, "reference", image_file)
        image_file = os.path.join(path, "image", image_file)

        assert os.path.exists(reference_file)

        # Read image and reference
        with rasterio.open(image_file) as image, rasterio.open(
            reference_file
        ) as reference:
            image_data: np.ndarray = image.read(1)
            reference_data: np.ndarray = reference.read(1)

            visualise_overlap(reference_data, image_data, image_file)


from utils import TrainImageDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    ds = TrainImageDataset("data/tmp/water")
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for image, reference in dl:
        print(image.shape, reference.shape)
        visualise_overlap(reference[0], image[0])
        break
