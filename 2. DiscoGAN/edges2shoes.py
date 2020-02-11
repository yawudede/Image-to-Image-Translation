import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Edges2Shoes(Dataset):
    def __init__(self, root, purpose='train'):

        self.files = sorted(glob.glob(os.path.join(root, purpose) + '/*.*'))
        self.transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.root = root
        self.purpose = purpose
        self.root = os.path.join(self.root, self.purpose)
        self.images = list(map(lambda x: os.path.join(self.root, x), os.listdir(self.root)))

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert("RGB")
        image = self.transform(image)

        w_total = image.size(2)
        w = int(w_total / 2)

        image_A = image[:, :64, :64]
        image_B = image[:, :64, w:w + 64]

        return {'A': image_A, 'B': image_B}

    def __len__(self):
        return len(self.images)


def get_edges2shoes_loader(purpose, batch_size):
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