import os
import random
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from config import *


class CelebA(Dataset):
    """CelebA Dataset"""
    def __init__(self, selected_attrs, purpose):

        self.image_path = './data/celeba/images/'
        self.attr_path = './data/celeba/list_attr_celeba.txt'
        self.selected_attrs = selected_attrs
        self.purpose = purpose

        self.train_dataset = []
        self.test_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.purpose = purpose
        self.transform = transforms.Compose([
            transforms.CenterCrop(config.crop_size),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.preprocess()

    def preprocess(self):
        """Preprocessing CelebA Dataset"""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr = lines[1].split()
        for i, attr in enumerate(all_attr):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        random.seed(1234)
        random.shuffle(lines[2:])

        for i, line in enumerate(lines[2:]):
            filename, *values = line.split()

            label = []
            for attr in self.selected_attrs:
                idx = self.attr2idx[attr]
                label.append(values[idx] == '1')

            if (i + 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Pre-processing CelebA Dataset is completed...')

    def __getitem__(self, index):
        dataset = self.train_dataset if self.purpose == 'train' else self.test_dataset

        image, label = dataset[index]
        image = Image.open(os.path.join(self.image_path, image))
        image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.train_dataset) if self.purpose == 'train' else len(self.test_dataset)


def get_celeba_loader(purpose, batch_size, selected_attrs):
    """CelebA Data Loader"""

    if purpose == 'train':
        train_celeba = CelebA(selected_attrs, purpose)
        train_celeba_loader = DataLoader(train_celeba, batch_size=batch_size, shuffle=True)
        return train_celeba_loader

    elif purpose == 'test':
        test_celeba = CelebA(selected_attrs, purpose)
        test_celeba_loader = DataLoader(test_celeba, batch_size=batch_size, shuffle=True)
        return test_celeba_loader

    else:
        raise NotImplementedError("Purpose should be either train or test.")
