from typing import Tuple
import torch
from torch import nn


class ConvNetClassifier(nn.Module):
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
            hidden_sizes=[1000, 500],
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)
        return x


def make_resnet(
    in_channels,
    num_layers,
    activation,
    kernel_size,
    dropout=0.0,
):
    layers = [
        nn.Conv2d(
            in_channels, 2 ** 5, kernel_size=kernel_size, padding=kernel_size // 2
        ),
        nn.Dropout(dropout),
        activation,
    ]
    layers += [
        # starts from in_channels = 32, out_channels = 64 and grows exponentially
        # with num_layers = 3 it goes to 256 channels
        resnet_block(
            2 ** (5 + i),
            2 ** (6 + i),
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
        )
        for i in range(num_layers)
    ]
    out_channels = 2 ** (6 + num_layers - 1)
    return nn.Sequential(*layers), out_channels


def mlp_block(in_size, out_size, activation, dropout=0.0):
    return nn.Sequential(nn.Linear(in_size, out_size), nn.Dropout(dropout), activation)


def make_mlp(in_size, out_size, hidden_sizes, activation, dropout=0.0):
    if len(hidden_sizes) == 0:
        return mlp_block(in_size, out_size, dropout=dropout, activation=activation)

    layers = []
    for hidden_size in hidden_sizes + [out_size]:
        layers.append(
            mlp_block(in_size, hidden_size, dropout=dropout, activation=activation)
        )
        in_size = hidden_size

    return nn.Sequential(*layers)


def resnet_block(in_channels, out_channels, activation, kernel_size, dropout=0.0):
    """1 residual convolution + 1 regular convolution + max pooling
    results in output width and height shrank by a factor of 2
    """
    return nn.Sequential(
        ResidualBlock(
            in_channels, kernel_size=kernel_size, activation=activation, dropout=dropout
        ),
        bottleneck_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
        ),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, activation=nn.Identity(), dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Dropout(dropout),
            activation,
        )

    def forward(self, x):
        return x + self.model(x)


def bottleneck_block(in_channels, out_channels, activation, kernel_size=3, dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        ),
        nn.Dropout(dropout),
        activation,
        nn.MaxPool2d(2),
    )


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
