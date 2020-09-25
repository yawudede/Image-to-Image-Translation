import os
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from config import *


class Edges2Shoes(Dataset):
    """Edges2Shoes Dataset"""
    def __init__(self, image_path, sort):
        self.path = os.path.join(image_path, sort)
        self.images = [x for x in sorted(os.listdir(self.path))]

        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
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


def get_edges2shoes_loader(purpose, batch_size):
    """Edges2Shoes Data Loader"""
    if purpose == 'train':
        train_A = Edges2Shoes('./data/edges2shoes/', 'trainA')
        train_B = Edges2Shoes('./data/edges2shoes/', 'trainB')

        train_loader_A = DataLoader(train_A, batch_size=batch_size, shuffle=True)
        train_loader_B = DataLoader(train_B, batch_size=batch_size, shuffle=True)

        return train_loader_A, train_loader_B

    elif purpose == 'test':
        test_A = Edges2Shoes('./data/edges2shoes/', 'testA')
        test_B = Edges2Shoes('./data/edges2shoes/', 'testB')

        test_loader_A = DataLoader(test_A, batch_size=batch_size, shuffle=True)
        test_loader_B = DataLoader(test_B, batch_size=batch_size, shuffle=True)

        return test_loader_A, test_loader_B

    else:
        raise NameError("Purpose should be either train or test.")