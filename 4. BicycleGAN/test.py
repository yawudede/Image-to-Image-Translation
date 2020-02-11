import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from models import *
from edges2handbags import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(batch_size):

    # Results Path #
    results_path = './data/results/generated/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Prepare Data Loader #
    val_loader = get_edges2handbags_loader('val', batch_size)
    total_batch = len(val_loader)

    # Constants #
    test_size = 20
    num_images = 5
    z_dim = 8

    # Prepare Generator 3
    G = Generator(z_dim).to(device)
    G.load_state_dict(torch.load('./data/results/BicycleGAN_Generator.pkl'))
    G.eval()

    # Fixed Noise #
    fixed_noise = Variable(torch.randn(test_size, num_images, z_dim).type(torch.FloatTensor)).to(device)

    # Test #
    print("Generating BicycleGAN with total batch of {}.".format(total_batch))
    for iters, (sketch, ground_truth) in enumerate(val_loader):

        # Prepare Data Loader #
        N = sketch.size(0)
        sketch = Variable(sketch.type(torch.FloatTensor)).to(device)
        results = torch.FloatTensor(N * (1 + num_images), 3, 128, 128)

        # Generate Fake Images #
        for i in range(N):
            results[i * (1 + num_images)] = sketch[i].data

            for j in range(num_images):
                image = sketch[i].unsqueeze(dim=0)
                noise_to_generator = fixed_noise[i, j, :].unsqueeze(dim=0)

                out = G(image, noise_to_generator)
                results[i * (1 + num_images) + j + 1] = out.data

                results = results / 2 + 0.5

        # Save Images #
        save_image(results, os.path.join(results_path, 'BicycleGAN_Edges2Handbags_Results_%03d.png'
                                         % (iters + 1)), nrow=(1 + num_images), normalize=True)

    make_gifs_test("BicycleGAN", results_path)


if __name__ == '__main__':
    batch_size = 1
    test(batch_size)