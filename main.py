from irisSegmentation import irisSegmentation
from preprocessing import preprocessing
from occlusionDetection import (
    upperEyelidDetection,
    lowerEyelidDetection,
    reflectionDetection,
)
from pathlib import Path
from tqdm.contrib.concurrent import process_map, thread_map
import hydra
from omegaconf import DictConfig
import cv2
import numpy as np

from typing import List


def parallelSegmentation(path: Path) -> None:
    # irisSegmentation(args[0], args[1], args[2])
    # preprocessing()
    normImg, img = irisSegmentation(path, config.dataset.color)
    cv2.imwrite(f"{config.folderNames.normPath}/norm_{path.name}", normImg)
    cv2.imwrite(f"{config.folderNames.dstPath}/{path.name}", img)

    upperMask = upperEyelidDetection(
        normImg[:, :, 2], numRays=46
    )  # red channel as input, numRays as in the haindl-krupicka paper
    lowerMask = lowerEyelidDetection(normImg[:, :, 2])  # red channel as input
    reflMask = reflectionDetection(normImg[:, :, 0])  # blue channel as input

    fullMask = np.zeros_like(normImg, dtype=np.uint8)
    # fullMask[lowerMask.astype(bool)] = 255
    # fullMask[upperMask.astype(bool)] = 255
    fullMask[upperMask.astype(bool) & lowerMask.astype(bool)] = 255
    fullMask[reflMask.astype(bool)] = 0

    cv2.imwrite(f"{config.folderNames.maskPath}/mask_{path.name}", fullMask)
    cv2.imwrite(f"{config.folderNames.upperPath}/upper_{path.name}", upperMask)
    cv2.imwrite(f"{config.folderNames.lowerPath}/lower_{path.name}", lowerMask)
    cv2.imwrite(f"{config.folderNames.reflPath}/refl_{path.name}", reflMask)


@hydra.main(config_name="config")
def loadConf(cfg: DictConfig) -> DictConfig:
    global config
    config = cfg


def mkdirs(paths: List[Path]):
    for path in paths:
        if not path.is_dir():
            path.mkdir()


if __name__ == "__main__":
    path = "."
    dir = Path(f"./{path}")
    with Path("./UTIRIS") as dataP:
        if not dataP.is_dir():
            from downloadDataset import downloadDataset

            utirisUrl = "http://www.dsp.utoronto.ca/~mhosseini/UTIRIS%20V.1.zip"
            downloadDataset(dataP, utirisUrl)

    loadConf()  # global config initialized

    pList = list(dir.glob(config.dataset.filePathsREGEX))
    color = config.dataset.color
    dst = Path(config.folderNames.dstPath)
    dirnames = [
        dst,
        Path(config.folderNames.normPath),
        Path(config.folderNames.maskPath),
        Path(config.folderNames.upperPath),
        Path(config.folderNames.lowerPath),
        Path(config.folderNames.reflPath),
    ]
    mkdirs(dirnames)

    print("\nmaking directories:", *dirnames, sep="\n - ")

    input("\n\npress enter to continue... (or ctrl+c to quit)")

    # for p in pList:
    # parallelSegmentation(p)

    # from tqdm.contrib.concurrent (basically is a Pool.imap with a tqdm.tqdm)
    process_map(parallelSegmentation, pList, max_workers=4)
