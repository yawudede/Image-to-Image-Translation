import os

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from config import *
from facades import get_facades_loader
from models import Generator
from utils import make_dirs, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_path)

    # Prepare Data Loader #
    test_loader = get_facades_loader('test', config.test_batch_size)

    # Prepare Generator #
    G = Generator().to(device)
    G.load_state_dict(torch.load(os.path.join(config.weights_path, 'Pix2Pix_Generator_Epochs_{}.pkl'.format(config.num_epochs))))
    G.eval()

    # Test #
    print("Pix2Pix | Generating facades images started...")
    for i, (input, target) in enumerate(test_loader):

        # Prepare Data #
        input = input.to(device)
        target = target.to(device)

        # Generate Fake Image #
        generated = G(input)

        # Save Images #
        result = torch.cat((target, input, generated), dim=0)
        save_image(result,
                   os.path.join(config.inference_path, 'Pix2Pix_Results_%03d.png' % (i+1)),
                   nrow=8,
                   normalize=True)

    make_gifs_test("Pix2Pix", config.inference_path)


if __name__ == '__main__':
    inference()