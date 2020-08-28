import os
from PIL import Image

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from config import *


class Edges2Handbags(Dataset):
    """Edges2Handbags Dataset"""
    def __init__(self, image_path, purpose):

        self.path = os.path.join(image_path, purpose)
        self.images = [x for x in sorted(os.listdir(self.path))]

        self.transform = transforms.Compose([
            transforms.Resize((config.crop_size, config.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.images[index])
        image = Image.open(image_path)
        width, height = image.size[0], image.size[1]

        sketch = image.crop((0, 0, int(width/2), height))
        target = image.crop((int(width/2), 0, width, height))

        sketch = self.transform(sketch)
        target = self.transform(target)

        return sketch, target

    def __len__(self):
        return len(self.images)


def get_edges2handbags_loader(purpose, batch_size):
    """Edges2Handbags Data Loader"""
    if purpose == 'train':
        train_set = Edges2Handbags('./data/edges2handbags/', 'train')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader

    elif purpose == 'val':
        val_set = Edges2Handbags('./data/edges2handbags/', 'val')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return val_loader

    else:
        raise NameError('Purpose should be train or val.')