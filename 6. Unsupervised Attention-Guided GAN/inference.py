import os

import torch
from torchvision.utils import save_image

from config import *
from horse2zebra import get_horse2zebra_loader
from models import Attention, Generator
from utils import make_dirs, denorm, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    paths = [config.inference_path_H2Z, config.inference_path_Z2H]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    test_horse_loader, test_zebra_loader = get_horse2zebra_loader('test', config.val_batch_size)

    # Prepare Attention and Generator #
    Attn_A = Attention().to(device)
    Attn_B = Attention().to(device)

    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    Attn_A.load_state_dict(torch.load(os.path.join(config.weights_path, 'UAG-GAN_Attention_A_Epoch_{}.pkl'.format(config.num_epochs))))
    Attn_B.load_state_dict(torch.load(os.path.join(config.weights_path, 'UAG-GAN_Attention_B_Epoch_{}.pkl'.format(config.num_epochs))))

    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'UAG-GAN_Generator_A2B_Epoch_{}.pkl'.format(config.num_epochs))))
    G_B2A.load_state_dict(torch.load(os.path.join(config.weights_path, 'UAG-GAN_Generator_B2A_Epoch_{}.pkl'.format(config.num_epochs))))

    # Test #
    print("UAG-GAN | Generating Horse2Zebra images started...")
    for i, (horse, zebra) in enumerate(zip(test_horse_loader, test_zebra_loader)):

        # Prepare Data #
        real_A = horse.to(device)
        real_B = zebra.to(device)

        # Generate Attention Images #
        attn_A = Attn_A(real_A.detach())
        attn_A = attn_A.repeat(1, 3, 1, 1)
        attn_A = 2 * attn_A - 1

        attn_B = Attn_B(real_B.detach())
        attn_B = attn_B.repeat(1, 3, 1, 1)
        attn_B = 2 * attn_B - 1

        # Generated Fake Images #
        fake_B = G_A2B(real_A.detach())
        fake_A = G_B2A(real_B.detach())

        # Save Images (Horse -> Zebra) #
        result = torch.cat((real_A, attn_A, fake_B), dim=0)
        save_image(denorm(result.data),
                   os.path.join(config.inference_path_H2Z, 'UAG-GAN_Horse2Zebra_Results_%03d.png' % (i+1))
                   )

        # Save Images (Zebra -> Horse) #
        result = torch.cat((real_B, attn_B, fake_A), dim=0)
        save_image(denorm(result.data),
                   os.path.join(config.inference_path_Z2H, 'UAG-GAN_Zebra2Horse_Results_%03d.png' % (i+1))
                   )

    # Make a GIF file #
    make_gifs_test("UAG-GAN", "Horse2Zebra", config.inference_path_H2Z)
    make_gifs_test("UAG-GAN", "Zebra2Horse", config.inference_path_Z2H)


if __name__ == '__main__':
    inference()