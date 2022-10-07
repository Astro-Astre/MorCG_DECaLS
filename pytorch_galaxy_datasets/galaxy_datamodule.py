from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_galaxy_datasets import galaxy_dataset
from torch.utils.data import Dataset


class GalaxyDataModule(Dataset):
    def __init__(
            self,
            batch_size=256,  # careful - will affect final performance
            num_workers=4,
            prefetch_factor=4,
            seed=1926
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transforms
        self.prefetch_factor = prefetch_factor
        self.dataloader_timeout = 240  # seconds

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0,
                          prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0,
                          prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0,
                          prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)
