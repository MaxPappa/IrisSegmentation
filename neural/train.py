import hydra
import omegaconf
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from neural.datamodule import IrisDataModule
from neural.lit_modules import ConvNetClassifier
from neural.project_utils import PROJECT_ROOT, log_hyperparameters


def parse_args():

    parser = pl.Trainer.add_argparse_args(ArgumentParser())

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_convnet_layers", default=5, type=int)
    parser.add_argument("--max_num_channels", default=256, type=int)
    parser.add_argument("--training_path", default="./results/", type=str)

    args = parser.parse_args()
    return args


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):

    # Instantiate all modules specified in the configs
    model = hydra.utils.instantiate(
        cfg.model,  # Object to instantiate
        _recursive_=True,
    )
    datamodule = hydra.utils.instantiate(cfg.data)
    callbacks = [
        # quando avremo il val set, questi diventano 'loss/val'
        pl.callbacks.ModelCheckpoint(monitor="loss/train"),
        pl.callbacks.EarlyStopping(monitor="loss/train", patience=2),
    ]

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(
        ".", "", "", log_graph=True, default_hp_metric=False
    )
    trainer = pl.Trainer(
        **omegaconf.OmegaConf.to_container(cfg.trainer),
        logger=tensorboard,
        callbacks=callbacks,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
