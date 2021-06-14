import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

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
        image_size,
        num_classes,
        num_convnet_layers=4,
        activation=nn.LeakyReLU(),
        dropout=0.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.num_convnet_layers = num_convnet_layers

        image_width, image_height = image_size
        convnet, out_channels = make_resnet(
            in_channels=3,
            num_layers=num_convnet_layers,
            dropout=dropout,
            activation=activation,
            kernel_size=3,
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
                num_classes=self.model.out_size,
                average="macro",
            )
            for metric_class in [Accuracy, Precision, Recall, F1]
        ]
        metrics += [
            ConfusionMatrix(
                num_classes=self.model.out_size,
                normalize="true",  # normalize over ground-truth labels
            )
        ]
        metrics = MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)
        return x

    #### LightningModule methods from now on:
    ####  - step
    ####  - training_step
    ####  - validation_step
    def step(self, batch):
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["label"].view(-1))
        return dict(loss=loss, logits=logits)

    def training_step(self, batch, batch_idx):
        step_result = self.step(batch)
        loss, logits = step_result["loss"], step_result["logits"]
        labels = batch["label"]

        metrics = self.train_metrics(logits, labels.view(-1))
        self.log_dict(metrics, on_epoch=True, on_step=False)

        self.log_dict(
            {"train/loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        step_result = self.step(batch)
        loss, logits = step_result["loss"], step_result["logits"]
        labels = batch["label"]

        metrics = self.val_metrics(logits, labels.view(-1))
        self.log_dict(metrics, on_epoch=True, on_step=False)

        self.log_dict(
            {"val/loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


if __name__ == "__main__":
    print("instantiating model ")
    model = ConvNetClassifier(
        image_size=(600, 100),
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
