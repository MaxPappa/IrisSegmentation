from copy import deepcopy
from random import sample
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

from dataclasses import dataclass
from typing import List, Tuple


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
    example_id: int
    image_path: Path
    mask_path: Path

    def __hash__(self):
        return sum(hash(x) for x in [self.subject_id, self.which_eye, self.example_id])


def get_dataloader(dataset_path, **dataloader_kwargs):
    dataset = IrisClassificationDataset(dataset_path)
    return DataLoader(dataset, **dataloader_kwargs)


from typing import Dict


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

        image_paths = {
            normalize_pathname(path): path for path in self.iris_path.glob("*.JPG")
        }
        mask_paths = {
            normalize_pathname(path): path for path in self.mask_path.glob("*.JPG")
        }

        # image_paths has to contain the same set of image names as mask_paths
        assert set(image_paths.keys()) == set(mask_paths.keys())

        # construct list of Item objects (our training data)
        self.data = []
        for normalized_pathname, image_path in image_paths.items():
            # example normalized_pathname: "001_L_1"
            # we care just about the first two fields separated by "_"
            subject_id, which_eye, example_id = normalized_pathname.split("_")
            subject_id, example_id = int(subject_id), int(example_id)
            mask_path = mask_paths[normalized_pathname]
            self.data.append(
                Item(
                    subject_id=subject_id - 1,
                    example_id=example_id - 1,
                    which_eye=which_eye,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
        self.data = sorted(
            self.data, key=lambda item: (item.subject_id, item.example_id)
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

    def train_test_split(
        dataset: "IrisClassificationDataset", val_percent: float
    ) -> Tuple["IrisClassificationDataset", "IrisClassificationDataset"]:

        train_data, val_data = dataset.random_fair_split(dataset, val_percent)

        train_dataset, val_dataset = deepcopy(dataset), deepcopy(dataset)

        train_dataset.data = train_data
        val_dataset.data = val_data

        return train_dataset, val_dataset

    @staticmethod
    def random_fair_split(dataset, val_percent: float):
        train, val = [], []
        # tutti gli occhi sinistri
        left_eyes = set(
            filter(
                lambda x: x.which_eye == "L",
                dataset.data,
            )
        )
        # tutti gli occhi destri
        right_eyes = set(
            filter(
                lambda x: x.which_eye == "R",
                dataset.data,
            )
        )

        new_data: List[Item] = []
        # per ciascun soggetto
        for subject_id in range(0, 80):
            # prendiamo tutti gli occhi sx di ciascun soggetto
            subject_left_eyes = list(
                filter(lambda item: item.subject_id == subject_id, left_eyes)
            )
            # prendiamo tutti gli occhi dx di ciascun soggetto
            subject_right_eyes = list(
                filter(lambda item: item.subject_id == subject_id, right_eyes)
            )

            # campioniamo (100-val_percent)% occhi sx per il training
            left_eyes_train = sample(
                subject_left_eyes,
                k=int((1 - val_percent) * len(subject_left_eyes)),
            )
            # campioniamo (100-val_percent)% occhi dx per il training
            right_eyes_train = sample(
                subject_right_eyes,
                k=int((1 - val_percent) * len(subject_right_eyes)),
            )
            # campioniamo val_percent% occhi dx per il validation
            # (escludendo quelli gia' campionati in precedenza per il training)
            right_eyes_val = sample(
                [i for i in subject_right_eyes if i not in right_eyes_train],
                k=int(val_percent * len(subject_right_eyes)),
            )
            # campioniamo val_percent% occhi sx per il validation
            left_eyes_val = sample(
                [i for i in subject_left_eyes if i not in left_eyes_train],
                k=int(val_percent * len(subject_left_eyes)),
            )
            train.extend(right_eyes_train)
            train.extend(left_eyes_train)
            val.extend(right_eyes_val)
            val.extend(left_eyes_val)
        return train, val


def normalize_pathname(path: Path) -> str:
    # example un-normalized path:
    #   ./path/to/PREFIX_IMG_001_L_1.JPG
    # normalizes to:
    #    IMG_001_L_1
    normalized_name = path.stem
    normalized_name = normalized_name[normalized_name.find("IMG_") + 4 :]
    return normalized_name


def normalize_pathnames(paths: List[Path]) -> List[str]:
    return [normalize_pathname(path) for path in paths]


# just some tests to see if it works
if __name__ == "__main__":
    dataloader = get_dataloader("./results/", batch_size=2)
    dataset = dataloader.dataset
    batch = next(iter(dataloader))

    val_pct = 0.2
    print(f"splitto il dataset con il {100*val_pct:.2f}% di validation set...")

    train_dataset, val_dataset = dataset.train_test_split(0.2)

    print(f"train: {len(train_dataset)} elementi")
    print(f"val: {len(val_dataset)} elementi")
