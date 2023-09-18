import os
import rasterio
import numpy as np
from torch.utils.data import Dataset
import torch


class TrainImageDataset(Dataset):
    def __init__(self, path: str):
        self.image_files = [
            os.path.join(path, "image", f)
            for f in os.listdir(os.path.join(path, "image"))
        ]
        self.reference_files = [
            os.path.join(path, "reference", os.path.basename(f))
            for f in self.image_files
        ]

        for ref in self.reference_files:
            assert os.path.exists(ref), f"Reference file {ref} doesn't exist."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        with rasterio.open(self.image_files[idx]) as image, rasterio.open(
            self.reference_files[idx]
        ) as reference:
            image_data: np.ndarray = image.read()
            reference_data: np.ndarray = reference.read()

            image_data = image_data.astype(np.float32)
            reference_data = reference_data.astype(np.float32)

            image_data = torch.tensor(image_data, dtype=torch.float32)
            reference_data = torch.tensor(reference_data, dtype=torch.float32)

            return image_data, reference_data


class TestImageDataset(Dataset):
    def __init__(self, path: str):
        self.input_files = [
            os.path.join(path, "input", f)
            for f in os.listdir(os.path.join(path, "input"))
        ]

        self.reference_files = [
            os.path.join(path, "reference", os.path.basename(f))
            for f in self.input_files
        ]

        self.mask_files = [
            os.path.join(path, "masks", os.path.basename(f)) for f in self.input_files
        ]

        self.tuw_results_files = [
            os.path.join(path, "tuw_results", os.path.basename(f))
            for f in self.input_files
        ]

        self.reference_files = [
            os.path.join(path, "reference", os.path.basename(f))
            for f in self.input_files
        ]

        for idx, ref in enumerate(self.reference_files):
            assert os.path.exists(
                self.reference_files[idx]
            ), f"Reference file {self.reference_files[idx]} doesn't exist."

            assert os.path.exists(
                self.mask_files[idx]
            ), f"Mask file {self.mask_files[idx]} doesn't exist."

            assert os.path.exists(
                self.tuw_results_files[idx]
            ), f"TuW results file {self.tuw_results_files[idx]} doesn't exist."

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        Returns:
            image_data: torch.Tensor
            reference_data: torch.Tensor
            mask_data: torch.Tensor
            tuw_results_data: torch.Tensor
        """
        with rasterio.open(self.input_files[idx]) as image, rasterio.open(
            self.reference_files[idx]
        ) as reference, rasterio.open(self.mask_files[idx]) as mask, rasterio.open(
            self.tuw_results_files[idx]
        ) as tuw_results:
            image_data: np.ndarray = image.read()
            reference_data: np.ndarray = reference.read()
            mask_data: np.ndarray = mask.read()
            tuw_results_data: np.ndarray = tuw_results.read()

            image_data = image_data.astype(np.float32)
            reference_data = reference_data.astype(np.float32)
            mask_data = mask_data.astype(np.float32)
            tuw_results_data = tuw_results_data.astype(np.float32)

            # Set 255 to 0 in tuw_results_data
            tuw_results_data[tuw_results_data == 255] = 0

            image_data = torch.tensor(image_data, dtype=torch.float32)
            reference_data = torch.tensor(reference_data, dtype=torch.float32)
            mask_data = torch.tensor(mask_data, dtype=torch.float32)
            tuw_results_data = torch.tensor(tuw_results_data, dtype=torch.float32)

            # Return filepath for debugging purposes

            return (
                image_data,
                reference_data,
                mask_data,
                tuw_results_data,
                self.input_files[idx],
            )
