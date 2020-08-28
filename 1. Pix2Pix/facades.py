import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import *


class Facades(Dataset):
    """Facades Dataset"""
    def __init__(self, image_dir, purpose):

        self.image_dir = image_dir
        self.purpose = purpose
        self.images = [x for x in sorted(glob(os.path.join(self.image_dir, self.purpose) + '/*.*'))]

        self.transform = transforms.Compose([
            transforms.Resize(config.crop_size, Image.BICUBIC),
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


def get_facades_loader(purpose, batch_size):
    """Facades Data Loader"""
    if purpose == 'train':
        train_set = Facades('./data/facades/', 'train')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader

    elif purpose == 'val':
        val_set = Facades('./data/facades/', 'val')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return val_loader

    elif purpose == 'test':
        test_set = Facades('./data/facades/', 'test')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return test_loader

    else:
        raise NameError('Purpose should be either train, val or test.')