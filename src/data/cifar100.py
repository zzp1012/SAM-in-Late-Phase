import os, torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple

# import internal libs
from utils import get_logger

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


def load(root: str = "../data",
         normalize: bool=True,
         randomize: bool=True,
         cutout: bool=False) -> Tuple[Dataset, Dataset]:
    """load the cifar100 dataset.
    Args:
        root (str): the root path of the dataset.
        normalize (bool): whether to normalize the dataset.
        randomize (bool): whether to randomize the dataset.
        cutout (bool): whether to apply cutout.
    Returns:
        return the dataset.
    """
    logger = get_logger(__name__)
    logger.info(f"""loading cifar100... with following settings:
        normalize: {normalize}; randomize: {randomize}; cutout: {cutout}""")

    # prepare the transform
    transform_train = transforms.Compose([
        *([transforms.RandomCrop(32, padding=4), 
           transforms.RandomHorizontalFlip()] if randomize else []), 
        transforms.ToTensor(),
        *([transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)] if normalize else []),
        *([Cutout()] if cutout else []),
    ])
    logger.info(f"transform_train: {transform_train}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        *([transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)] if normalize else []),
    ])
    logger.info(f"transform_test: {transform_test}")

    # load the dataset
    os.makedirs(root, exist_ok=True)
    trainset = datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test)

    # show basic info of dataset
    logger.info(f"trainset size: {len(trainset)}")
    logger.info(f"testset size: {len(testset)}")
    return trainset, testset