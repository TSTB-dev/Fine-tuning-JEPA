import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import zipfile
import tarfile
import lmdb
import pickle

import torch
from PIL import Image

class CUB200Dataset(Dataset):
    img_base_url = "https://data.caltech.edu/records/65de6-vp158/files/"
    anno_base_url = "https://data.caltech.edu/records/w9d68-gec53/files/"
    files = {
        "images": "CUB_200_2011.tgz",
        "annotations": "segmentations.tgz"
    }
    query = "?download=1"
    base_folder = "CUB_200_2011/images"

    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "cub")
        self.transform = transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
        self.load_data()

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        for key, filename in self.files.items():
            if filename == "CUB_200_2011.tgz":
                url = self.img_base_url + filename + self.query
                print(f"Donwloading from {url}...")
                download_url(url, self.root, filename)
                self.extract_file(os.path.join(self.root, filename))
            elif filename == "segmentations.tgz":
                url = self.anno_base_url + filename + self.query
                print(f"Donwloading from {url}...")
                download_url(url, self.root, filename)
                self.extract_file(os.path.join(self.root, filename))
            else:
                raise ValueError("Invalid filename")

    def extract_file(self, file_path):
        if file_path.endswith("tar.gz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)
        elif file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
        elif file_path.endswith("tgz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)
        else:
            raise ValueError("Invalid file extension")
    
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True
    
    def load_data(self):
        images = []
        labels = []
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            images.append(filepath)
            labels.append(row.target)
        
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        
        image = default_loader(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "index": index,
            "image": image,
            "label": label,
            "path": image_path,
        }

class CUB200DatasetLMDB(Dataset):
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
        self.num_classes = 200
        
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
            
            # For cub dataset, original label is [200], so we need to convert it to [199]
            label = torch.tensor(sample["label"] - 1)
            
            return {
                "index": index,
                "image": image,
                "label": label,
                "caption": str(sample["label"]),
                "path": sample["path"]
            }
    
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from src.datasets.util import worker_init_fn
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = CUB200DatasetLMDB(root="data/cub200.lmdb", download=True, train=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, worker_init_fn=worker_init_fn)
    # Check num classes
    classes = set()
    for batch in dataloader:
        classes.update(batch["label"].numpy())
    
    print(f"Num classes: {len(classes)}")
    print(f"Classes: {classes}")