import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import zipfile
import tarfile
import numpy as np
import lmdb
import pickle

from torchvision.datasets.utils import download_url

class PetsDataset(Dataset):
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    files = {
        "images": "images.tar.gz",
        "annotations": "annotations.tar.gz"
    }

    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "pets")
        self.transform = transform
        self.train = train

        if download:
            self.download()

        self.images, self.labels = self.load_data()

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        for key, filename in self.files.items():
            url = self.base_url + filename
            download_url(url, self.root, filename)
            self.extract_file(os.path.join(self.root, filename))

    def extract_file(self, file_path):
        if file_path.endswith("tar.gz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)
        elif file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

    def load_data(self):
        if self.train:
            annotations_path = os.path.join(self.root, "annotations", "trainval.txt")
        else:
            annotations_path = os.path.join(self.root, "annotations", "test.txt")
        
        images = []
        labels = []
        with open(annotations_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                image_name = parts[0]
                label = int(parts[1]) - 1
                images.append(os.path.join(self.root, "images", image_name + ".jpg"))
                labels.append(label)
                
        # set number of classes
        self.classes = sorted(set(labels))
        self.num_classes = len(set(labels))
        
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": str(label),
            "path": image_path
        }

class PetsDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.lmdb_path = root
        self.transform = transform
        self.train = train
        self.split = 'train' if train else 'test'
        self.env = None
        
        try:
            with lmdb.open(root, readonly=True, lock=False) as env:
                with env.begin() as txn:
                    self.length = len([key for key, _ in txn.cursor() if key.decode('ascii').startswith(self.split)])
        except Exception as e:
            raise IOError(f"Failed to open LMDB at {root}")
        
        self.num_classes = 37
    
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
    dataset = PetsDataset(root="data", download=True, train=False)
    print(len(dataset))
    # print(dataset[0]["image"].size)
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from src.datasets.util import save_to_lmdb
    save_to_lmdb(PetsDataset(root="data", download=True), "data/pets.lmdb")
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = PetsDatasetLMDB("data/pets.lmdb", transform=trans, train=False)
    print(f"Number of samples: {len(dataset)}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    for batch in dataloader:
        print(batch["image"].size())
        break
    
    