import os

import torch
from torchvision.utils import save_image

from config import *
from selfie2anime import get_selfie2anime_loader
from models import Generator
from utils import denorm, make_dirs, make_gifs_test

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_path)

    # Prepare Data Loader #
    test_loader_selfie, test_loader_anime = get_selfie2anime_loader('test', config.batch_size)

    # Prepare Generator #
    G_A2B = Generator(image_size=config.crop_size, num_blocks=config.num_blocks).to(device)

    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'U-GAT-IT_G_A2B_Epoch_{}.pkl'.format(config.num_epochs))))

    # Inference #
    print("U-GAT-IT | Generating Selfie2Anime images started...")
    with torch.no_grad():
        for i, (selfie, anime) in enumerate(zip(test_loader_selfie, test_loader_anime)):

            # Prepare Data #
            real_A = selfie.to(device)

            # Generate Fake Images #
            fake_B = G_A2B(real_A)[0]

            # Save Images (Selfie -> Anime) #
            result = torch.cat((real_A, fake_B), dim=0)
            save_image(denorm(result.data),
                       os.path.join(config.inference_path, 'U-GAT-IT_Selfie2Anime_Results_%03d.png' % (i + 1))
                       )

    # Make a GIF file #
    make_gifs_test("U-GAT-IT", "Selfie2Anime", config.inference_path)


if __name__ == '__main__':
    inference()