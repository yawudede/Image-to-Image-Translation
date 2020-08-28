import os
import torch
from torchvision.utils import save_image

from config import *
from edges2handbags import get_edges2handbags_loader
from models import Generator
from utils import make_dirs, denorm, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_path)

    # Prepare Data Loader #
    val_loader = get_edges2handbags_loader('val', config.val_batch_size)

    # Prepare Generator #
    G = Generator(z_dim=config.z_dim).to(device)
    G.load_state_dict(torch.load(os.path.join(config.weights_path, 'BicycleGAN_Generator_Epoch_{}.pkl'.format(config.num_epochs))))
    G.eval()

    # Fixed Noise #
    fixed_noise = torch.randn(config.test_size, config.num_images, config.z_dim).to(device)

    # Test #
    print("BiCycleGAN | Generating Edges2Handbags Images started...")
    for iters, (sketch, ground_truth) in enumerate(val_loader):

        # Prepare Data #
        N = sketch.size(0)
        sketch = sketch.to(device)
        results = torch.FloatTensor(N * (1 + config.num_images), 3, config.crop_size, config.crop_size)

        # Generate Fake Images #
        for i in range(N):
            results[i * (1 + config.num_images)] = sketch[i].data

            for j in range(config.num_images):
                image = sketch[i].unsqueeze(dim=0)
                noise_to_generator = fixed_noise[i, j, :].unsqueeze(dim=0)

                out = G(image, noise_to_generator)
                results[i * (1 + config.num_images) + j + 1] = out

            # Save Images #
            save_image(denorm(results.data),
                       os.path.join(config.inference_path, 'BicycleGAN_Edges2Handbags_Results_%03d.png' % (iters + 1)),
                       nrow=(1 + config.num_images),
                       )

    make_gifs_test("BicycleGAN", config.inference_path)


if __name__ == '__main__':
    inference()