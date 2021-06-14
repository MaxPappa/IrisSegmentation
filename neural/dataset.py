import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

from dataclasses import dataclass
from typing import List


@dataclass
class Item:
    """
    For use internal to the IrisClassificationDataset class.
    Holds together information about a single training example:
        subject_id : int
            which subject this iris belongs to (also the classification target)
        which_eye : str
            either 'L' or 'R', indicates left or right eye
        image_path : pathlib.Path
            path to the normalized iris image
        mask_path : pathlib.Path
            path to the occlusion mask image
    """

    subject_id: int
    which_eye: str
    image_path: Path
    mask_path: Path


def get_dataloader(dataset_path, **dataloader_kwargs):
    dataset = IrisClassificationDataset(dataset_path)
    return DataLoader(dataset, **dataloader_kwargs)


class IrisClassificationDataset(Dataset):
    """
    Contains examples of kind (image, mask, label)
    where
        image is the normalized iris
        mask a binary mask about occlusions
        label is the numerical ID of the subject
    """

    def __init__(self, path: str):
        self.base_path = Path(path)
        self.iris_path = self.base_path / "normalized"
        self.mask_path = self.base_path / "mask"
        assert (
            self.base_path.is_dir()
            and self.iris_path.is_dir()
            and self.mask_path.is_dir()
        )

        self.image_paths = {
            normalize_pathname(path): path for path in self.iris_path.glob("*.JPG")
        }
        self.mask_paths = {
            normalize_pathname(path): path for path in self.mask_path.glob("*.JPG")
        }

        # self.image_paths has to contain the same set of image names as self.mask_paths
        assert set(self.image_paths.keys()) == set(self.mask_paths.keys())

        # construct list of Item objects (our training data)
        self.data = []
        for normalized_pathname, image_path in self.image_paths.items():
            # example normalized_pathname: "IMG_001_L_1.JPG"
            # we care just about the first two fields separated by "_"
            _, subject_id, which_eye, *_ = normalized_pathname.split("_")
            subject_id = int(subject_id)
            mask_path = self.mask_paths[normalized_pathname]
            self.data.append(
                Item(
                    subject_id=subject_id - 1,
                    which_eye=which_eye,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

        self.image_transforms = T.Compose(
            [
                T.ToTensor(),  # from [h,w,c] in range(0,255) to [c,h,w] in range(0,1)
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.mask_transforms = T.Compose(
            [
                T.ToTensor(),  # from [h,w,c] in range(0,255) to [c,h,w] in range(0,1)
            ]
        )

    def __getitem__(self, idx):
        """Retrieve one Item instance from self.data and construct dict for training examples"""
        item = self.data[idx]
        label = item.subject_id
        image = Image.open(item.image_path).convert("RGB")
        mask = Image.open(item.mask_path).convert("L")  # to grayscale image

        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)

        return dict(image=image, mask=mask, label=label)

    def __len__(self):
        return len(self.data)


def normalize_pathname(path: Path) -> str:
    # example un-normalized path:
    #   ./path/to/PREFIX_IMG_001_L_1.JPG
    # normalizes to:
    #    IMG_001_L_1.JPG
    return path.stem[path.stem.find("_") + 1 :]


def normalize_pathnames(paths: List[Path]) -> List[str]:
    return [normalize_pathname(path) for path in paths]


# just some tests to see if it works
if __name__ == "__main__":
    dataloader = get_dataloader("./results/", batch_size=2)
    dataset = dataloader.dataset
    batch = next(iter(dataloader))
