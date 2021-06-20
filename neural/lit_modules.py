from math import log2
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from typing import Tuple
import hydra
from omegaconf import DictConfig
from neural.project_utils import PROJECT_ROOT

from torchsummary import summary

from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1,
    ConfusionMatrix,
)

from neural.models import make_resnet, make_mlp


class Classifier(pl.LightningModule):
    def __init__(
        self,
        name: str,  # symbolic name (used in logs)
        classifier: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.classifier = classifier
        self.num_classes = self.classifier.get_num_classes()
        self.weight_decay = weight_decay
        # used by pytorch lightning to extimate some stuff
        self.example_input_array = self.classifier.example_input_array

        self.loss_criterion = nn.CrossEntropyLoss()

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

    def forward(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

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

        metrics = self.train_metrics(preds, labels.view(-1))

        self.log_dict(
            {"loss/train": loss, **metrics},
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

        metrics = self.val_metrics(preds, labels.view(-1))

        self.log_dict(
            {"loss/val": loss, **metrics},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    # def training_epoch_end(self, step_outputs):
    #     metrics = self.train_metrics.compute()
    #     self.log_dict(metrics)

    # def validation_epoch_end(self, step_outputs):
    #     train_metrics = self.train_metrics.compute()
    #     val_metrics = self.val_metrics.compute()

    #     self.log_dict(train_metrics)
    #     self.log_dict(val_metrics)

    #     for train_metric_key, train_metric in train_metrics.items():
    #         _, metric_name = train_metric_key.split("/")

    #         val_metric_key = train_metric_key.replace("train", "val")
    #         val_metric = val_metrics[val_metric_key]
    #         self.logger.experiment.add_scalars(
    #             metric_name,
    #             {train_metric_key: train_metric, val_metric_key: val_metric},
    #             global_step=self.global_step,
    #         )


class UntrainedConvNet(pl.LightningModule):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        num_classes,
        num_convnet_layers=4,
        activation=nn.LeakyReLU(),
        dropout=0.0,
        min_num_channels=16,
        max_num_channels=256,
        stride=1,
        mlp_hidden_dim=512,
    ):
        super().__init__()

        min_exponent = log2(min_num_channels)
        # controlla che min_exponent sia un numero intero
        # (e che quindi min_num_channels era una potenza di 2)
        assert (
            min_exponent - round(min_exponent) == 0
        ), "You must provide a power of 2 as the minimum number of convnet channels"
        min_exponent = int(min_exponent)

        self.save_hyperparameters()
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.num_convnet_layers = num_convnet_layers
        self.max_num_channels = max_num_channels
        self.mlp_hidden_dim = mlp_hidden_dim
        self.stride = stride
        self.min_exponent = min_exponent

        self.input_shape = (3, image_width, image_height)
        self.example_input_array = torch.randn(self.input_shape).unsqueeze(
            0
        )  # nice things for pytorch lightning

        convnet, out_channels = make_resnet(
            in_channels=3,
            num_layers=num_convnet_layers,
            dropout=dropout,
            activation=activation,
            kernel_size=3,
            max_num_channels=max_num_channels,
            stride=self.stride,
            min_exponent=self.min_exponent,
        )
        self.conv_out_channels = out_channels
        self.convnet = nn.Sequential(
            convnet,
        )
        self.flattener = nn.Flatten(1)  # converts conv output to shape [batch, -1]

        with torch.no_grad():
            conv_output = self.flattener(self.convnet(self.example_input_array))
            flattened_size = conv_output.numel()
        assert flattened_size != 0, "Convolution results in a size 0 feature map!"

        # classifier just gets a flattened view of image
        self.classifier = make_mlp(
            flattened_size,
            num_classes,
            hidden_sizes=[mlp_hidden_dim],
            dropout=dropout,
            activation=activation,
        )

        hydra.utils.log(
            summary(self, self.input_shape, batch_size=-1, device=str(self.device))
        )

    def forward(self, x):
        feature_map = self.convnet(x)
        flattened = self.flattener(feature_map)
        logits = self.classifier(flattened)
        return logits

    def get_num_classes(self):
        return self.num_classes


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    print("instantiating model ")
    model = hydra.utils.instantiate(cfg.model)
    print("model created")

    from neural.dataset import get_dataloader

    dataloader = get_dataloader(dataset_path="./results/", batch_size=4)

    batch = next(iter(dataloader))

    x, y = batch["image"], batch["label"]

    model_output = model(x)
    loss = nn.functional.cross_entropy(model_output, y)
    loss.backward()


if __name__ == "__main__":
    main()
