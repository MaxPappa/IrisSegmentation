import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from typing import Tuple

from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1,
    ConfusionMatrix,
)

from neural.models import make_resnet, make_mlp


class ConvNetClassifier(pl.LightningModule):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        num_classes,
        name: str = "untrained_convnet",  # symbolic name (used in logs)
        num_convnet_layers=4,
        activation=nn.LeakyReLU(),
        dropout=0.0,
        learning_rate=1e-3,
        max_num_channels=256,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.num_convnet_layers = num_convnet_layers
        self.lr = learning_rate
        self.loss_criterion = nn.CrossEntropyLoss()

        convnet, out_channels = make_resnet(
            in_channels=3,
            num_layers=num_convnet_layers,
            dropout=dropout,
            activation=activation,
            kernel_size=3,
            max_num_channels=max_num_channels,
        )
        self.convnet = nn.Sequential(
            convnet,
            nn.Flatten(1),  # [batch, -1]
        )
        flattened_size = out_channels * (
            (image_width // (2 ** num_convnet_layers))
            * (image_height // (2 ** num_convnet_layers))
        )

        self.classifier = make_mlp(
            flattened_size,
            num_classes,
            hidden_sizes=[1000],
            dropout=dropout,
            activation=activation,
        )

        metrics = [
            metric_class(
                num_classes=self.num_classes + 1,
                average="micro",
            )
            for metric_class in [Accuracy, Precision, Recall, F1]
        ]
        metrics = MetricCollection(metrics)
        self.train_metrics = metrics.clone(postfix="/train")
        self.val_metrics = metrics.clone(postfix="/val")

        self.example_input_array = torch.randn((1, 3, image_width, image_height))

    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        logits = self(x)
        return self._compute_predictions(logits)

    def _compute_predictions(self, logits):
        probabs = torch.log_softmax(logits, dim=-1)
        preds = probabs.argmax(-1)
        return preds

    #### LightningModule methods from now on:
    ####  - step
    ####  - training_step
    ####  - validation_step
    ####  - configure_optimizers

    def step(self, batch):
        logits = self(batch["image"])
        loss = self.loss_criterion(logits, batch["label"].view(-1))
        return dict(loss=loss, logits=logits)

    def training_step(self, batch, batch_idx):
        step_result = self.step(batch)
        loss, logits = step_result["loss"], step_result["logits"]
        labels = batch["label"]
        preds = self._compute_predictions(logits)

        self.train_metrics(preds, labels.view(-1))

        self.log_dict(
            {"loss/train": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        step_result = self.step(batch)
        loss, logits = step_result["loss"], step_result["logits"]
        labels = batch["label"]
        preds = self._compute_predictions(logits)

        self.val_metrics(preds, labels.view(-1))

        self.log_dict(
            {"loss/val": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def on_train_start(self):
    #     # Proper logging of hyperparams and metrics in TB
    #     val_metrics = self.val_metrics.compute()
    #     self.logger.log_hyperparams(
    #         self.hparams,
    #         {
    #             "loss/val": 1.0,
    #             **val_metrics,
    #         },
    #     )

    def training_epoch_end(self, step_outputs):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

    def validation_epoch_end(self, step_outputs):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)


if __name__ == "__main__":
    print("instantiating model ")
    model = ConvNetClassifier(
        image_width=600,
        image_height=100,
        num_classes=79,
        dropout=0.1,
    )
    print("model created")

    from neural.dataset import get_dataloader

    dataloader = get_dataloader(dataset_path="./results/", batch_size=4)

    batch = next(iter(dataloader))

    x, y = batch["image"], batch["label"]

    model_output = model(x)
    loss = nn.functional.cross_entropy(model_output, y)
    loss.backward()
    print(f"{loss=}")
