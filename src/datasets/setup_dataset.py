"""
Download and convert the dataset to LMDB format.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import argparse
from tqdm import tqdm
import pickle
from src.datasets.util import save_to_lmdb

from src.datasets.pets import PetsDataset, PetsDatasetLMDB
from src.datasets.caltech import Caltech101Dataset, Caltech101DatasetLMDB
from src.datasets.flowers import Flowers102Dataset, Flowers102DatasetLMDB
from src.datasets.cars import StanfordCarsDataset, StanfordCarsDatasetLMDB
from src.datasets.cub import CUB200Dataset, CUB200DatasetLMDB

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str,
        help="root directory of dataset",
        default="data"
    )
    parser.add_argument(
        "--dataset_name", type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="download the dataset",
    )
    args = parser.parse_args()
    
    if "pets" in args.dataset_name:
        train_dataset = PetsDataset(root=args.root, download=args.download, train=True)
        save_to_lmdb(train_dataset, os.path.join(args.root, "pets.lmdb"), split="train")
        logger.info(f"Saved pets dataset to LMDB: {args.root}/pets.lmdb")
        test_dataset = PetsDataset(root=args.root, download=args.download, train=False)
        save_to_lmdb(test_dataset, os.path.join(args.root, "pets.lmdb"), split="test")
        logger.info(f"Saved pets dataset to LMDB: {args.root}/pets.lmdb")
        try:
            train_dataset_lmdb = PetsDatasetLMDB(os.path.join(args.root, "pets.lmdb"))
            test_dataset_lmdb = PetsDatasetLMDB(os.path.join(args.root, "pets.lmdb"), train=False)
            logger.info(f"Succesfully loaded pets dataset from LMDB: {args.root}/pets.lmdb")
        except Exception as e:
            logger.error(f"Failed to load pets dataset from LMDB: {args.root}/pets.lmdb")
            logger.error(e)
            
    elif "caltech" in args.dataset_name:
        train_dataset = Caltech101Dataset(root=args.root, download=args.download, train=True)
        save_to_lmdb(train_dataset, os.path.join(args.root, "caltech101.lmdb"), split="train")
        logger.info(f"Saved caltech101 dataset to LMDB: {args.root}/caltech101.lmdb")
        test_dataset = Caltech101Dataset(root=args.root, download=args.download, train=False)
        save_to_lmdb(test_dataset, os.path.join(args.root, "caltech101.lmdb"), split="test")
        logger.info(f"Saved caltech101 dataset to LMDB: {args.root}/caltech101.lmdb")
        try:
            train_dataset_lmdb = Caltech101DatasetLMDB(os.path.join(args.root, "caltech101.lmdb"))
            test_dataset_lmdb = Caltech101DatasetLMDB(os.path.join(args.root, "caltech101.lmdb"), train=False)
            logger.info(f"Succesfully loaded caltech101 dataset from LMDB: {args.root}/caltech101.lmdb")
        except Exception as e:
            logger.error(f"Failed to load caltech101 dataset from LMDB: {args.root}/caltech101.lmdb")
            logger.error(e)
    
    elif "flowers" in args.dataset_name:
        train_dataset = Flowers102Dataset(root=args.root, download=args.download, train=True)
        save_to_lmdb(train_dataset, os.path.join(args.root, "flowers102.lmdb"), split="train")
        logger.info(f"Saved flowers102 dataset to LMDB: {args.root}/flowers102.lmdb")
        test_dataset = Flowers102Dataset(root=args.root, download=args.download, train=False)
        save_to_lmdb(test_dataset, os.path.join(args.root, "flowers102.lmdb"), split="test")
        logger.info(f"Saved flowers102 dataset to LMDB: {args.root}/flowers102.lmdb")
        try:
            train_dataset_lmdb = Flowers102DatasetLMDB(os.path.join(args.root, "flowers102.lmdb"))
            test_dataset_lmdb = Flowers102DatasetLMDB(os.path.join(args.root, "flowers102.lmdb"), train=False)
            logger.info(f"Succesfully loaded flowers102 dataset from LMDB: {args.root}/flowers102.lmdb")
        except Exception as e:
            logger.error(f"Failed to load flowers102 dataset from LMDB: {args.root}/flowers102.lmdb")
            logger.error(e)
    
    elif "cars" in args.dataset_name:
        train_dataset = StanfordCarsDataset(root=args.root, download=args.download, train=True)
        save_to_lmdb(train_dataset, os.path.join(args.root, "stanford_cars.lmdb"), split="train")
        logger.info(f"Saved stanford_cars dataset to LMDB: {args.root}/stanford_cars.lmdb")
        test_dataset = StanfordCarsDataset(root=args.root, download=args.download, train=False)
        save_to_lmdb(test_dataset, os.path.join(args.root, "stanford_cars.lmdb"), split="test")
        logger.info(f"Saved stanford_cars dataset to LMDB: {args.root}/stanford_cars.lmdb")
        try:
            train_dataset_lmdb = StanfordCarsDatasetLMDB(os.path.join(args.root, "stanford_cars.lmdb"))
            test_dataset_lmdb = StanfordCarsDatasetLMDB(os.path.join(args.root, "stanford_cars.lmdb"), train=False)
            logger.info(f"Succesfully loaded stanford_cars dataset from LMDB: {args.root}/stanford_cars.lmdb")
        except Exception as e:
            logger.error(f"Failed to load stanford_cars dataset from LMDB: {args.root}/stanford_cars.lmdb")
            logger.error(e)
    
    elif "cub" in args.dataset_name:
        train_dataset = CUB200Dataset(root=args.root, download=args.download, train=True)
        save_to_lmdb(train_dataset, os.path.join(args.root, "cub200.lmdb"), split="train")
        logger.info(f"Saved cub200 dataset to LMDB: {args.root}/cub200.lmdb")
        test_dataset = CUB200Dataset(root=args.root, download=args.download, train=False)
        save_to_lmdb(test_dataset, os.path.join(args.root, "cub200.lmdb"), split="test")
        logger.info(f"Saved cub200 dataset to LMDB: {args.root}/cub200.lmdb")
        try:
            train_dataset_lmdb = CUB200DatasetLMDB(os.path.join(args.root, "cub200.lmdb"))
            test_dataset_lmdb = CUB200DatasetLMDB(os.path.join(args.root, "cub200.lmdb"), train=False)
            logger.info(f"Succesfully loaded cub200 dataset from LMDB: {args.root}/cub200.lmdb")
        except Exception as e:
            logger.error(f"Failed to load cub200 dataset from LMDB: {args.root}/cub200.lmdb")
            logger.error(e)
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
if __name__ == "__main__":
    main()
        
    