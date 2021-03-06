from typing import Tuple
import torch
from torch import nn


def make_resnet(
    in_channels,
    num_layers,
    activation,
    kernel_size,
    max_num_channels=512,
    dropout=0.0,
    stride=1,
    min_exponent=4,  # number of conv filters increases exponentially from a layer to the next
):
    out_exponent = min_exponent + 1
    layers = [
        nn.Conv2d(
            in_channels,
            2 ** min_exponent,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
        ),
        nn.BatchNorm2d(2 ** min_exponent),
        nn.Dropout(dropout),
        activation,
    ]
    layers += [
        # starts from in_channels = 32, out_channels = 64 and grows exponentially
        # with num_layers = 3 it goes to 256 channels
        resnet_block(
            min(max_num_channels, 2 ** (min_exponent + i)),
            min(max_num_channels, 2 ** (out_exponent + i)),
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
            stride=stride,
        )
        for i in range(num_layers)
    ]
    out_channels = min(max_num_channels, 2 ** (out_exponent + num_layers - 1))
    return nn.Sequential(*layers), out_channels


def mlp_block(in_size, out_size, activation, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.BatchNorm1d(out_size),
        nn.Dropout(dropout),
        activation,
    )


def make_mlp(in_size, out_size, hidden_sizes, activation, dropout=0.0):
    if len(hidden_sizes) == 0:
        return mlp_block(in_size, out_size, dropout=dropout, activation=activation)

    layers = []
    for i, hidden_size in enumerate(hidden_sizes + [out_size]):
        # if last iteration,
        # append output layer and quit
        if i == len(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            break  # might have been continue as well, it is the last iteration for sure
        layers.append(
            mlp_block(in_size, hidden_size, dropout=dropout, activation=activation)
        )
        in_size = hidden_size

    return nn.Sequential(*layers)


def resnet_block(
    in_channels, out_channels, activation, kernel_size, dropout=0.0, stride=1
):
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
            stride=stride,
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
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
            activation,
        )

    def forward(self, x):
        return x + self.model(x)


def bottleneck_block(
    in_channels, out_channels, activation, kernel_size=3, dropout=0.0, stride=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
        ),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(dropout),
        activation,
        nn.MaxPool2d(2),
    )
