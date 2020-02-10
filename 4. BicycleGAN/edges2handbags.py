from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image

"""https://arxiv.org/pdf/1711.11586.pdf"""


class Edges2Handbags(Dataset):
    def __init__(self, root, purpose):

        self.root = root
        self.purpose = purpose
        self.images = os.listdir(os.path.join(self.root, self.purpose))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.purpose, self.images[index])
        image = Image.open(image_path)
        width, height = image.size[0], image.size[1]

        sketch = image.crop((0, 0, int(width/2), height))
        ground_truth = image.crop((int(width/2), 0, width, height))

        sketch = self.transform(sketch)
        ground_truth = self.transform(ground_truth)

        return (sketch, ground_truth)

    def __len__(self):
        return len(self.images)


def get_edges2handbags(purpose, batch_size):
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