import os
import tarfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import scipy.io
import lmdb
import pickle

class Flowers102Dataset(Dataset):
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    files = {
        "images": "102flowers.tgz",
        "labels": "imagelabels.mat",
        "splits": "setid.mat"
    }

    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "flowers")
        self.transform = transform
        self.split = 'train' if train else 'test'

        if download:
            self.download()

        self.images, self.labels = self.load_data()

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        for key, filename in self.files.items():
            url = self.base_url + filename
            download_url(url, self.root, filename)
            if filename.endswith(".tgz"):
                self.extract_file(os.path.join(self.root, filename))

    def extract_file(self, file_path):
        if file_path.endswith("tgz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)

    def load_data(self):
        # Load labels
        labels_path = os.path.join(self.root, self.files['labels'])
        labels = scipy.io.loadmat(labels_path)['labels'][0] - 1

        # Load splits
        splits_path = os.path.join(self.root, self.files['splits'])
        splits = scipy.io.loadmat(splits_path)
        if self.split == 'train':
            indices = np.concatenate((splits['trnid'][0], splits['valid'][0])) - 1
        else:
            indices = splits['tstid'][0] - 1

        images = []
        selected_labels = []
        images_dir = os.path.join(self.root, "jpg")
        for idx in indices:
            image_name = f"image_{idx + 1:05d}.jpg"
            images.append(os.path.join(images_dir, image_name))
            selected_labels.append(labels[idx])

        self.classes = sorted(set(selected_labels))
        self.num_classes = len(self.classes)
        
        return images, selected_labels
    
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
    
class Flowers102DatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=False, download=False):
        self.lmdb_path = root
        self.transform = transform
        self.split = "train" if train else "test"
        
        self.env = None

        try:
            with lmdb.open(root, readonly=True, lock=False) as env:
                with env.begin() as txn:
                    self.length = len([key for key, _ in txn.cursor() if key.decode('ascii').startswith(self.split)])
        except Exception as e:
            raise IOError(f"Failed to open LMDB at {root}")
        
        self.num_classes = 102
    
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
    dataset = Flowers102Dataset(root="data", download=True, train=True)
    print(f"Flowers102 dataset successfully loaded.")
    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes}")
    print(f"Sample image shape: {dataset[0]['image'].size()}")
    print(f"Sample label: {dataset[0]['label']}")
    
