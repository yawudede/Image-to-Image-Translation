import os
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from config import *


class Horse2Zebra(Dataset):
    """Horses2Zebra Dataset"""
    def __init__(self, image_path, sort):
        self.path = os.path.join(image_path, sort)
        self.images = [x for x in sorted(os.listdir(self.path))]

        self.transform = transforms.Compose([
            transforms.Resize(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.images[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


def get_horse2zebra_loader(purpose, batch_size):
    """Horses2Zebra Data Loader"""
    if purpose == 'train':
        train_horse = Horse2Zebra('./data/horse2zebra/', 'trainA')
        train_zebra = Horse2Zebra('./data/horse2zebra/', 'trainB')

        trainloader_horse = DataLoader(train_horse, batch_size=batch_size, shuffle=True)
        trainloader_zebra = DataLoader(train_zebra, batch_size=batch_size, shuffle=True)

        return trainloader_horse, trainloader_zebra

    elif purpose == 'test':
        test_horse = Horse2Zebra('./data/horse2zebra/', 'testA')
        test_zebra = Horse2Zebra('./data/horse2zebra/', 'testB')

        testloader_horse = DataLoader(test_horse, batch_size=batch_size, shuffle=True)
        testloader_zebra = DataLoader(test_zebra, batch_size=batch_size, shuffle=True)

        return testloader_horse, testloader_zebra

    else:
        raise NameError("Purpose should be either train or test.")