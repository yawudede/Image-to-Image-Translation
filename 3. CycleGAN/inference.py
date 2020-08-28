import os

import torch
from torchvision.utils import save_image

from config import *
from horse2zebra import get_horse2zebra_loader
from models import Generator
from utils import make_dirs, denorm, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    paths = [config.inference_path_H2Z, config.inference_path_Z2H]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    test_horse_loader, test_zebra_loader = get_horse2zebra_loader('test', config.val_batch_size)

    # Prepare Generator #
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'CycleGAN_Generator_A2B_Epoch_{}.pkl'.format(config.num_epochs))))
    G_B2A.load_state_dict(torch.load(os.path.join(config.weights_path, 'CycleGAN_Generator_B2A_Epoch_{}.pkl'.format(config.num_epochs))))

    # Test #
    print("CycleGAN | Generating Horse2Zebra images started...")
    for i, (horse, zebra) in enumerate(zip(test_horse_loader, test_zebra_loader)):

        # Prepare Data #
        real_A = horse.to(device)
        real_B = zebra.to(device)

        # Generate Fake Images #
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        # Generated Reconstructed Images #
        fake_ABA = G_B2A(fake_B)
        fake_BAB = G_A2B(fake_A)

        # Save Images (Horse -> Zebra) #
        result = torch.cat((real_A, fake_B, fake_ABA), dim=0)
        save_image(denorm(result.data),
                   os.path.join(config.inference_path_H2Z, 'CycleGAN_Horse2Zebra_Results_%03d.png' % (i+1)),
                   nrow=3,
                   normalize=True)

        # Save Images (Zebra -> Horse) #
        result = torch.cat((real_B, fake_A, fake_BAB), dim=0)
        save_image(denorm(result.data),
                   os.path.join(config.inference_path_Z2H, 'CycleGAN_Zebra2Horse_Results_%03d.png' % (i+1)),
                   nrow=3,
                   normalize=True)

    # Make a GIF file #
    make_gifs_test("CycleGAN", "Horse2Zebra", config.inference_path_H2Z)
    make_gifs_test("CycleGAN", "Zebra2Horse", config.inference_path_Z2H)


if __name__ == '__main__':
    inference()