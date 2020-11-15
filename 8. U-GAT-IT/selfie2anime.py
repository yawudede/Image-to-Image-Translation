from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image

from config import *


class Selfie2Anime(Dataset):
    """Selfie2Anime Dataset"""
    def __init__(self, images_path, sort):
        self.sort = sort
        self.images_path = os.path.join(images_path, sort)
        self.images = [x for x in sorted(os.listdir(self.images_path))]

        self.train_transform = transforms.Compose([
            transforms.Resize(config.crop_size + 30),
            transforms.RandomCrop(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.images[index])
        image = Image.open(image_path).convert('RGB')
        if self.sort == 'trainA' or 'trainB':
            image = self.train_transform(image)
        elif self.sort == 'testA' or 'testB':
            image = self.test_transform(image)
        return image

    def __len__(self):
        return len(self.images)


def get_selfie2anime_loader(purpose, batch_size):
    """Prepare Data Loader"""
    # A is Selfie, B is Anime #

    if purpose == 'train':
        train_selfie = Selfie2Anime('./selfie2anime/', 'trainA')
        train_anime = Selfie2Anime('./selfie2anime/', 'trainB')

        train_loader_selfie = DataLoader(train_selfie, batch_size=batch_size, shuffle=True)
        train_loader_anime = DataLoader(train_anime, batch_size=batch_size, shuffle=True)

        return train_loader_selfie, train_loader_anime

    elif purpose == 'test':
        test_selfie = Selfie2Anime('./selfie2anime/', 'testA')
        test_anime = Selfie2Anime('./selfie2anime/', 'testB')

        test_loader_selfie = DataLoader(test_selfie, batch_size=batch_size, shuffle=True)
        test_loader_anime = DataLoader(test_anime, batch_size=batch_size, shuffle=True)

        return test_loader_selfie, test_loader_anime