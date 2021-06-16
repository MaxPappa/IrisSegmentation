from typing import Optional
import pytorch_lightning as pl
import torch

from neural.dataset import get_dataloader, IrisClassificationDataset


class IrisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        num_workers: int,
        name: str = "iris_dataset",  # symbolic name (for logging)
        val_percent: float = 0.2,  # size of val split (float in [0,1])
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent

    # first-time setup
    def setup(self, stage: Optional[str] = None):
        dataset = IrisClassificationDataset(self.dataset_path)

        self.train_dataset, self.val_dataset = dataset.train_test_split(
            self.val_percent
        )
        print(
            f"Split fatto, train: {len(self.train_dataset)} val: {len(self.val_dataset)}"
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
