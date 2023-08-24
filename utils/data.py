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
