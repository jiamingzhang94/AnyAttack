import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class EvalDataset(Dataset):
    def __init__(self, data_list, image_dir):
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取caption
        captions = item['caption']

        return {
            'images': image,
            'captions': captions
        }

def filter_dataset(dataset):
    filtered_annotations = [ann for ann in dataset.annotation if ann['label'] != 'entailment']
    dataset.annotation = filtered_annotations
    return dataset

def train_collate_fn(batch):
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
    ])

    images = []
    captions = []
    for item in batch:
        image = item['image']
        # caption = item['text_input']

        # 转换图像
        if isinstance(image, Image.Image):
            image = transform(image)

        images.append(image)
        # captions.append(caption)

    # 将图像堆叠成一个批次
    images = torch.stack(images)

    return {
        'image': images,
        'caption': None
    }


def eval_collate_fn(batch):
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
    ])

    images = []
    captions = []
    for item in batch:
        image = item['image']
        caption = item['text_input']

        # 转换图像
        if isinstance(image, Image.Image):
            image = transform(image)

        images.append(image)
        captions.append(caption)

    # 将图像堆叠成一个批次
    images = torch.stack(images)

    return {
        'image': images,
        'caption': captions
    }


def custom_collate(batch):
    images = torch.stack([item['images'] for item in batch])
    captions = [item['captions'] for item in batch]

    # 获取每个样本的caption数量
    caption_lengths = [len(c) for c in captions]

    return {
        'images': images,
        'captions': captions,
        'caption_lengths': caption_lengths
    }
