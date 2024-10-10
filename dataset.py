import torch
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.datasets import ImageFolder
import requests
from io import BytesIO
import argparse
import os
import tarfile
import io
from torchvision import transforms
import matplotlib.pyplot as plt
import json

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

class ImageTextDataset(Dataset):
    def __init__(self, it_pair_path, image_path, image_only=False, transform=None):
        with open(it_pair_path, 'r', encoding='utf-8') as f:
            self.it_pair = json.load(f)
        self.transform = transform
        self.image_path = image_path
        self.image_only = image_only

    def __len__(self):
        return len(self.it_pair)

    def __getitem__(self, idx):
        sample = self.it_pair[idx]

        image_path = os.path.join(self.image_path, sample['image'])
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        try:
            text = sample['caption'][0]
        except:
            text = None
        return image, text



class SubsetImageFolder(Dataset):
    def __init__(self, root, samples_per_class=5, transform=None):
        self.dataset = ImageFolder(root, transform=transform)
        self.samples_per_class = samples_per_class
        self.class_to_samples = self._get_class_samples()

        self.subset_samples = []
        for class_idx, samples in self.class_to_samples.items():
            self.subset_samples.extend(samples[:samples_per_class])

    def _get_class_samples(self):
        class_to_samples = {}
        for idx, (path, class_idx) in enumerate(self.dataset.samples):
            if class_idx not in class_to_samples:
                class_to_samples[class_idx] = []
            class_to_samples[class_idx].append((path, class_idx))

        # Sort samples within each class to ensure consistency
        for class_idx in class_to_samples:
            class_to_samples[class_idx].sort()

        return class_to_samples

    def __len__(self):
        return len(self.subset_samples)

    def __getitem__(self, idx):
        path, class_idx = self.subset_samples[idx]
        sample = self.dataset.loader(path)
        if self.dataset.transform is not None:
            sample = self.dataset.transform(sample)
        return sample, class_idx

