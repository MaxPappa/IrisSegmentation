from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from neural.datamodule import IrisDataModule
from neural.lit_modules import ConvNetClassifier


def parse_args():

    parser = pl.Trainer.add_argparse_args(ArgumentParser())

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_convnet_layers", default=5, type=int)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    trainer = pl.Trainer.from_argparse_args(args)

    model = ConvNetClassifier(
        (600, 100),
        79,
        num_convnet_layers=args.num_convnet_layers,
        activation=torch.nn.LeakyReLU(),
        dropout=0.2,
        learning_rate=args.learning_rate,
    )
    datamodule = IrisDataModule(training_path="./results/", batch_size=args.batch_size)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
