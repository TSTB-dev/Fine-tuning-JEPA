import os
import tarfile
import torch
import subprocess
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from src.datasets.imagenet_1k_classes import IMAGENET2012_CLASSES
# from imagenet_1k_classes import IMAGENET2012_CLASSES

import logging
logger = logging.getLogger(__name__)

class ImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.split = "train" if train else "val"  # 'train', 'val', or 'test'
        self.transform = transform
        self.access_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        assert self.access_token is not None, "Please set the HUGGINGFACE_TOKEN environment variable."
        
        self.data_dir = os.path.join(self.root, f"{self.split}_images")

        if download:
            self.download_archives()
            self.extract_archives()

        self.image_paths, self.labels, self.captions = self.load_image_paths_and_labels()
        self.num_classes = len(set(self.labels))

    def download_archives(self):
        base_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"
        headers = f"Authorization: Bearer {self.access_token}"

        if self.split == 'train':
            archives = [f"train_images_{i}.tar.gz" for i in range(5)]
        elif self.split == 'val':
            archives = ["val_images.tar.gz"]
        elif self.split == 'test':
            archives = ["test_images.tar.gz"]
        else:
            raise ValueError("Invalid split. Expected 'train', 'val' or 'test'.")

        for archive in archives:
            full_url = f"{base_url}/{archive}"
            archive_path = os.path.join(self.root, "data", archive)

            if not os.path.exists(archive_path):
                logger.info(f"Downloading {archive} from {full_url}...")
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                subprocess.run(["wget", "--header", headers, "-O", archive_path, full_url], check=True)
            else:
                logger.info(f"{archive} already exists, skipping download.")

    def extract_archives(self):
        archive_path = os.path.join(self.root, "data")
        
        if self.split == 'train':
            archives = [f"train_images_{i}.tar.gz" for i in range(5)]
        elif self.split == 'val':
            archives = ["val_images.tar.gz"]
        elif self.split == 'test':
            archives = ["test_images.tar.gz"]
        else:
            raise ValueError("Invalid split. Expected 'train', 'val' or 'test'.")

        for archive in archives:
            full_path = os.path.join(archive_path, archive)
            if os.path.exists(full_path):
                with tarfile.open(full_path, 'r:gz') as tar:
                    logger.info(f"Extracting {archive}...")
                    tar.extractall(path=self.data_dir)
            else:
                raise FileNotFoundError(f"{full_path} not found. Please download it first.")

    def load_image_paths_and_labels(self):
        image_paths = []
        labels = []
        captions = []
        # data_dir contains the images with the following structure:
        # - train_images
        #   - {classid}_{imageid}_{classid}.JPEG
        # - val_images
        #   - ILSVRC2012_val_{imageid}_{classid}.JPEG
        
        data_dir = Path(self.data_dir)
        img_files = list(data_dir.glob("*.JPEG"))
        for img_file in img_files:
            image_paths.append(str(img_file))
            if self.split == "train":
                class_id = img_file.stem.split("_")[0]  # classid_imageid_classid.JPEG => classid
            elif self.split == "val":
                class_id = img_file.stem.split("_")[-1]  # ILSVRC2012_val_imageid_classid.JPEG => classid
            else:
                raise ValueError("Invalid split. Expected 'train', 'val' or 'test'.")
            try:
                class_caption = IMAGENET2012_CLASSES[class_id]
            except:
                logger.warning(f"Class ID {class_id} not found in IMAGENET2012_CLASSES.")
                logger.warning(f"Skipping {img_file}.")
                continue
            class_index = list(IMAGENET2012_CLASSES.keys()).index(class_id)
            labels.append(class_index)
            captions.append(class_caption)

        return image_paths, labels, captions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        caption = self.captions[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "caption": caption,
            "path": image_path
        }
        
class ImageNetSubsetDataset(ImageNetDataset):
    def __init__(self, root, train=True, transform=None, download=False, num_classes=1000, num_samples_per_class=100, random_extract=False):
        super().__init__(root, train, transform, download)
        self.num_classes = num_classes
        self.random_extract = random_extract
        self.num_samples_per_class = num_samples_per_class
        self.image_paths, self.labels, self.captions = self.load_image_paths_and_labels()
        
        print(f"Extracting subset with {self.num_samples_per_class} samples per class...")
        self.extract_subset()
        logger.info(f"Extracted subset with {len(self)} images.")
        
        self.num_classes = len(set(self.labels))
    
    def extract_subset(self):
        subset_image_paths = []
        subset_labels = []
        subset_captions = []
        logger.info(f"Extracting {self.num_samples_per_class} samples per class...")
        
        for class_index in range(self.num_classes):
            class_image_paths = [path for path, label in zip(self.image_paths, self.labels) if label == class_index]
            if self.random_extract:
                import random
                class_image_paths = random.sample(class_image_paths
                                                    , self.num_samples_per_class)
            else:
                class_image_paths = class_image_paths[:self.num_samples_per_class]
        
            subset_image_paths.extend(class_image_paths)
            subset_labels.extend([class_index]*len(class_image_paths))
            subset_captions.extend([self.captions[class_index]]*len(class_image_paths))
        
        self.image_paths = subset_image_paths
        self.labels = subset_labels
        self.captions = subset_captions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        caption = self.captions[index]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image: {image_path}")
            logger.error(e)
            return None
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "caption": caption,
            "path": image_path
        }
    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    root = "/home/haselab/projects/sakai/Fine-tuning-JEPA/data/imagenet"  
    # dataset = ImageNetDataset(root=root, train=False, transform=transform, download=False)
    # print(f"ImageNet dataset successfully loaded.")
    # print(f"Number of images: {len(dataset)}")
    # print(f"Sample data: {dataset[0]}")
    
    dataset = ImageNetSubsetDataset(root=root, train=True, transform=transform, download=False)
    print(f"ImageNetSubsetDataset successfully loaded.")
    print(f"Number of images: {len(dataset)}")
    print(f"Sample data: {dataset[0]}")
    print(f"Sample image shape: {dataset[0]['image'].size()}")
    print(f"Sample label: {dataset[0]['label']}")