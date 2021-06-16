import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from neural.project_utils import PROJECT_ROOT, log_hyperparameters

logger = logging.getLogger(__name__)


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
def main(cfg: DictConfig):
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Instantiate all modules specified in the configs
    model = hydra.utils.instantiate(
        cfg.model,  # Object to instantiate
        _recursive_=True,
    )

    datamodule = hydra.utils.instantiate(cfg.data)

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(
        ".", "", "", log_graph=False, default_hp_metric=False
    )
    # "./lightning_logs/", "default", None, log_graph=True, default_hp_metric=False

    callbacks = [
        # quando avremo il val set, questi diventano 'loss/val'
        pl.callbacks.ModelCheckpoint(monitor=cfg.train.monitor_metric),
        pl.callbacks.EarlyStopping(
            monitor=cfg.train.monitor_metric,
            patience=cfg.train.early_stop.patience,
            mode=cfg.train.early_stop.mode,
        ),
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        logger=tensorboard,
        callbacks=callbacks,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
