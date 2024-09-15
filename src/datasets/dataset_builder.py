from .caltech import Caltech101Dataset, Caltech101DatasetLMDB
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .cub import CUB200Dataset
from .flowers import Flowers102Dataset, Flowers102DatasetLMDB
from .pets import PetsDataset, PetsDatasetLMDB
from .cars import StanfordCarsDataset, StanfordCarsDatasetLMDB

import torch

__all__ = [
    "Caltech101Dataset",
    "Caltech101DatasetLMDB",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "CUB200Dataset",
    "Flowers102Dataset",
    "Flowers102DatasetLMDB",
    "PetsDataset",
    "PetsDatasetLMDB",
    "StanfordCarsDataset",
    "StanfordCarsDatasetLMDB",
]

def make_dataset(name, **kwargs):
    if name == "caltech101":
        return Caltech101Dataset(**kwargs)
    elif name == "cifar10":
        return CIFAR10Dataset(**kwargs)
    elif name == "cifar100":
        return CIFAR100Dataset(**kwargs)
    elif name == "cub200":
        return CUB200Dataset(**kwargs)
    elif name == "flowers102":
        return Flowers102Dataset(**kwargs)
    elif name == "pets":
        return PetsDataset(**kwargs)
    elif name == "stanford_cars":
        return StanfordCarsDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")