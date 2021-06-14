from typing import Optional
import pytorch_lightning as pl
import torch

from neural.dataset import get_dataloader


class IrisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_path: str,
        batch_size: int,
        num_workers: int,
        name: str = "iris_dataset",  # symbolic name (for logging)
    ):
        super().__init__()
        self.training_path = training_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return get_dataloader(
            self.training_path,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    # to be defined...
    # def val_dataloader(self):
    #     pass

    # may be useful to make first-time setup
    # def setup(self, stage: Optional[str] = None):
    #     pass
