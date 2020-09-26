import os

import torch
from torchvision.utils import save_image

from config import *
from edges2shoes import get_edges2shoes_loader
from models import AdaIN_Generator
from utils import make_dirs, denorm, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_random_path)
    make_dirs(config.inference_ex_guided_path)

    # Prepare Data Loader #
    test_loader_A, test_loader_B = get_edges2shoes_loader('test', config.val_batch_size)

    # Prepare Generator #
    G_A2B = AdaIN_Generator().to(device)
    G_B2A = AdaIN_Generator().to(device)

    G_A2B.load_state_dict(torch.load(os.path.join(config.weights_path, 'MUNIT_Generator_A2B_Epoch_{}.pkl'.format(config.num_epochs))))
    G_B2A.load_state_dict(torch.load(os.path.join(config.weights_path, 'MUNIT_Generator_B2A_Epoch_{}.pkl'.format(config.num_epochs))))

    G_A2B.eval()
    G_B2A.eval()

    # Test #
    print("MUNIT | Generating Edges2Shoes images started...")

    for i, (real_A, real_B) in enumerate(zip(test_loader_A, test_loader_B)):

        # Prepare Data #
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        if config.style == "Random":
            random_style = torch.randn(real_A.size(0), config.style_dim, 1, 1).to(device)
            style = random_style
            results = [real_A]

        elif config.style == "Ex_Guided":
            _, style = G_A2B.encode(real_B)
            results = [real_A, real_B]

        else:
            raise NotImplementedError

        for j in range(config.num_inference):

            content, _ = G_B2A.encode(real_A[j].unsqueeze(dim=0))
            results.append(G_A2B.decode(content, style[j].unsqueeze(0)))

            # Save Images #
            result = torch.cat(results, dim=0)

            if config.style == "Random":
                title='MUNIT_Edges2Shoes_%s_Results_%03d.png' % (config.style, i+1)
                path = os.path.join(config.inference_random_path, title)

            elif config.style == "Ex_Guided":
                title='MUNIT_Edges2Shoes_%s_Results_%03d.png' % (config.style, i+1)
                path = os.path.join(config.inference_ex_guided_path, title)

            else:
                raise NotImplementedError

            save_image(result.data,
                       path,
                       nrow=config.num_inference,
                       normalize=True)

    # Make a GIF file #
    if config.style == "Random":
        make_gifs_test("MUNIT", config.style, config.inference_random_path)

    elif config.style == "Ex_Guided":
        make_gifs_test("MUNIT", config.style, config.inference_ex_guided_path)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    inference()