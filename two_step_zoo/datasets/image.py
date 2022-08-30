import os
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms

from two_step_zoo.datasets.supervised_dataset import SupervisedDataset


class CelebA(Dataset):
    """
    CelebA PyTorch dataset
    The built-in PyTorch dataset for CelebA is outdated.
    """

    def __init__(self, root: str, role: str = "train"):
        self.root = Path(root)
        self.role = role

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        celeb_path = lambda x: self.root / x

        role_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        splits_df = pd.read_csv(celeb_path("list_eval_partition.csv"))
        self.filename = splits_df[splits_df["partition"] == role_map[self.role]]["image_id"].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = (self.root / "img_align_celeba" /
                    "img_align_celeba" / self.filename[index])
        X = PIL.Image.open(img_path)
        X = self.transform(X)

        return X, 0

    def __len__(self) -> int:
        return len(self.filename)


def get_image_datasets_by_class(dataset_name, data_root, valid_fraction):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "celeba":
        # valid_fraction ignored
        data_class = CelebA

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_dset = data_class(root=data_dir, role="train")
    valid_dset = data_class(root=data_dir, role="valid")
    test_dset = data_class(root=data_dir, role="test")

    return train_dset, valid_dset, test_dset


def image_tensors_to_dataset(dataset_name, dataset_role, images, labels):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    return SupervisedDataset(dataset_name, dataset_role, images, labels)


# Returns tuple of form `(images, labels)`. Both are uint8 tensors.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 255}
def get_raw_image_tensors(dataset_name, train, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return images.to(torch.uint8), labels.to(torch.uint8)


def get_torchvision_datasets(dataset_name, data_root, valid_fraction):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)

    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]

    train_dset = image_tensors_to_dataset(dataset_name, "train", train_images, train_labels)
    valid_dset = image_tensors_to_dataset(dataset_name, "valid", valid_images, valid_labels)

    test_images, test_labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    test_dset = image_tensors_to_dataset(dataset_name, "test", test_images, test_labels)

    return train_dset, valid_dset, test_dset


def get_image_datasets(dataset_name, data_root, make_valid_dset):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.1 if make_valid_dset else 0

    torchvision_datasets = ["mnist", "fashion-mnist", "svhn", "cifar10"]

    get_datasets_fn = get_torchvision_datasets if dataset_name in torchvision_datasets else get_image_datasets_by_class

    return get_datasets_fn(dataset_name, data_root, valid_fraction)
