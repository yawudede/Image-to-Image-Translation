import os
import torch
from torchvision.utils import save_image

from config import *
from celeba import get_celeba_loader
from models import Generator
from utils import make_dirs, denorm, make_gifs_test, create_labels

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_path)

    # Prepare Data Loader #
    test_loader = get_celeba_loader('test', config.batch_size, config.selected_attrs)

    # Prepare Generator #
    G = Generator(num_classes=len(config.selected_attrs)).to(device)
    G.load_state_dict(torch.load(os.path.join(config.weights_path, 'StarGAN_Generator_Epoch_{}.pkl'.format(config.num_epochs))))

    # Test #
    print("StarGAN | Generating Aligned CelebA Images started...")
    for i, (image, label) in enumerate(test_loader):

        # Prepare Data #
        image = image.to(device)
        fixed_labels = create_labels(label, selected_attrs=config.selected_attrs)

        # Generate Fake Images #
        x_fake_list = [image]

        for c_fixed in fixed_labels:
            x_fake_list.append(G(image, c_fixed))
        x_concat = torch.cat(x_fake_list, dim=3)

        # Save Images #
        save_image(denorm(x_concat.data.cpu()),
                   os.path.join(config.inference_path, 'StarGAN_Aligned_CelebA_Results_%04d.png' % (i + 1)),
                   nrow=1,
                   padding=0)

    make_gifs_test("StarGAN", config.inference_path)


if __name__ == '__main__':
    inference()
