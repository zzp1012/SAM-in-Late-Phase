from torch.utils.data import Dataset
from typing import Tuple

def prepare_dataset(dataset: str,
                    root: str = "../../data/",
                    **kwargs) -> Tuple[Dataset, Dataset]:
    """prepare the dataset.

    Args:
        dataset (str): the dataset name.
        root (str): the root path of the dataset.

    Returns:
        trainset and testset
    """
    if dataset == "cifar10":
        import data.cifar10 as cifar10
        trainset, testset = cifar10.load(root, **kwargs)
    elif dataset == "cifar100":
        import data.cifar100 as cifar100
        trainset, testset = cifar100.load(root, **kwargs)
    else:
        raise NotImplementedError(f"dataset {dataset} is not implemented.")
    return trainset, testset