from .caltech import Caltech101Dataset, Caltech101DatasetLMDB
from .cub import CUB200Dataset, CUB200DatasetLMDB
from .flowers import Flowers102Dataset, Flowers102DatasetLMDB
from .pets import PetsDataset, PetsDatasetLMDB
from .cars import StanfordCarsDataset, StanfordCarsDatasetLMDB

import torch
import lmdb
from tqdm import tqdm
import pickle

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.env = lmdb.open(dataset.lmdb_path, readonly=True, lock=False)

def make_dataset(name, is_lmdb: bool = False, **kwargs):
    if name == "caltech101":
        return Caltech101Dataset(**kwargs) if not is_lmdb else Caltech101DatasetLMDB(**kwargs)
    elif name == "cub200":
        return CUB200Dataset(**kwargs) if not is_lmdb else CUB200DatasetLMDB(**kwargs)
    elif name == "flowers102":
        return Flowers102Dataset(**kwargs) if not is_lmdb else Flowers102DatasetLMDB(**kwargs)
    elif name == "pets":
        return PetsDataset(**kwargs) if not is_lmdb else PetsDatasetLMDB(**kwargs)
    elif name == "stanford_cars":
        return StanfordCarsDataset(**kwargs) if not is_lmdb else StanfordCarsDatasetLMDB(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    
def save_to_lmdb(dataset, lmdb_path, split, map_size=1e12):
    """Convert a dataset to LMDB format and save it to disk.
    Args:
        dataset: Dataset to save to LMDB.
        lmdb_path: Path to save the LMDB database.
        map_size: Maximum size of the LMDB database.
    """
    logger.info(f"Saving dataset to LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, map_size=int(map_size))
    
    with env.begin(write=True) as txn:
        logging.info(f"Converting dataset to LMDB: {lmdb_path}")
        for i in tqdm(range(len(dataset))):
            try:
                sample = dataset[i]
                image = sample["image"]
                label = sample["label"].item() if isinstance(sample["label"], torch.Tensor) else sample["label"]
                path = sample["path"]
                
                image_bytes = image.tobytes()
                image_size = image.size
                
                key = f"{split}_{i}".encode("ascii")
                
                txn.put(
                    key,
                    pickle.dumps({
                        "image": image_bytes,
                        "label": label,
                        "path": path,
                        "size": image_size
                    })
                )
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue
    env.close()
    logger.info(f"Dataset saved to LMDB: {lmdb_path}")
    