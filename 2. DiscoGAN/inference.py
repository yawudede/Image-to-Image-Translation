import os

import torch
from torchvision.utils import save_image

from config import *
from edges2shoes import get_edges2shoes_loader
from models import Generator
from utils import make_dirs, denorm, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_path)

    # Prepare Data Loader #
    val_loader = get_edges2shoes_loader(purpose='val', batch_size=config.val_batch_size)

    # Prepare Generator #
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'DiscoGAN_Generator_A2B_Epoch_{}.pkl'.format(config.num_epochs))))
    G_B2A.load_state_dict(torch.load(os.path.join(config.weights_path, 'DiscoGAN_Generator_B2A_Epoch_{}.pkl'.format(config.num_epochs))))

    # Test #
    print("DiscoGAN | Generating Edges2Shoes images started...")
    for i, (real_A, real_B) in enumerate(val_loader):

        # Prepare Data #
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Generate Fake Images #
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        # Generated Reconstructed Images #
        fake_ABA = G_B2A(fake_B)
        fake_BAB = G_A2B(fake_A)

        # Save Images #
        result = torch.cat((real_A, fake_A, fake_BAB, real_B, fake_B, fake_ABA), dim=0)
        save_image(denorm(result.data),
                   os.path.join(config.inference_path, 'DiscoGAN_Edges2Shoes_Results_%03d.png' % (i+1)),
                   nrow=8,
                   normalize=True)

    # Make a GIF file #
    make_gifs_test("DiscoGAN", config.inference_path)


if __name__ == '__main__':
    inference()