import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
import tarfile
import h5py
import numpy as np

from torchvision.datasets.utils import download_url

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision
from tqdm import tqdm
import lmdb
import pickle

from dotenv import load_dotenv
load_dotenv()
import kaggle

class StanfordCarsDataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = root
        self.transform = transform
        self.train = train

        if download:
            self.download()

        # Use torchvision's StanfordCars dataset class
        split = 'train' if self.train else 'test'
        self.dataset = torchvision.datasets.StanfordCars(root=self.root, split=split, transform=self.transform, download=False)
        print(f"Number of samples in {split} set: {len(self.dataset)}")
        
        # Access the data and labels from the torchvision dataset
        # self.images = []
        # self.labels = []
        # for i in tqdm(range(len(self.dataset))):
        #     self.images.append(self.dataset[i][0])
        #     self.labels.append(self.dataset[i][1])
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

    def download(self):
        if os.path.exists(os.path.join(self.root, "stanford_cars", "devkit")):
            print("Dataset already downloaded")
            return
        # Download and unzip the dataset using Kaggle API:
        kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path=self.root, unzip=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": str(label),
            "path": self.dataset._samples[index][0]  # Get the image path
        }

class StanfordCarsDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.lmdb_path = root
        self.transform = transform
        self.train = train
        self.split = 'train' if train else 'test'
        self.num_classes = 196
        
        self.env = None
        
        try:
            with lmdb.open(root, readonly=True, lock=False) as env:
                with env.begin() as txn:
                    self.length = len([key for key, _ in txn.cursor() if key.decode('ascii').startswith(self.split)])
        except Exception as e:
            raise IOError(f"Failed to open LMDB at {root}")
    
    def __len__(self):
        return self.length
           
    def __getitem__(self, index):
        key = f"{self.split}_{index}".encode("ascii")
        with self.env.begin() as txn:
            data = txn.get(key)
            assert data is not None, f"Failed to retrieve key {key}"
            sample = pickle.loads(data)
            image = Image.frombytes("RGB", sample["size"], sample["image"])
            
            if self.transform:
                image = self.transform(image)
                
            label = torch.tensor(sample["label"])
            
            return {
                "index": index,
                "image": image,
                "label": label,
                "caption": str(sample["label"]),
                "path": sample["path"]
            }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = StanfordCarsDataset(root="data", transform=transform, download=True, train=True)
    print(len(dataset))
    print(dataset[0]["image"])
    print(dataset[0]["label"])
    print(dataset[0]["path"])
