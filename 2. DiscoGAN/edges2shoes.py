import os
from glob import glob
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from config import *

class Edges2Shoes(Dataset):
    """Edges2Shoes Dataset"""
    def __init__(self, image_dir, purpose):

        self.image_dir = image_dir
        self.purpose = purpose
        self.images = [x for x in sorted(glob(os.path.join(self.image_dir, self.purpose) + '/*.*'))]

        self.transform = transforms.Compose([
            transforms.Resize(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):

        image = Image.open(self.images[index])
        width, height = image.size

        image_A = image.crop((width/2, 0, width, height))
        image_B = image.crop((0, 0, width / 2, height))

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)

        return (image_A, image_B)

    def __len__(self):
        return len(self.images)


def get_edges2shoes_loader(purpose, batch_size):
    """Edges2Shoes Data Loader"""
    if purpose == 'train':
        train_set = Edges2Shoes('./data/edges2shoes/', 'train')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_loader

    elif purpose == 'val':
        test_set = Edges2Shoes('./data/edges2shoes/', 'val')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        return test_loader

    else:
        raise NameError("Purpose should be either train or val.")