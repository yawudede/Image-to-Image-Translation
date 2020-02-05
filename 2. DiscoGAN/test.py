import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from itertools import chain

from models import *
from edges2shoes import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(batch_size):

    results_path = './data/results/generated/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    val_loader = get_edges2shoes_loader(batch_size=batch_size, purpose='val')
    total_batch = len(val_loader)

    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    G_A2B.load_state_dict(torch.load('./data/results/DiscoGAN_Generator_A2B.pkl'))
    G_B2A.load_state_dict(torch.load('./data/results/DiscoGAN_Generator_B2A.pkl'))

    G_A2B.eval()
    G_B2A.eval()

    print("Generating DiscoGAN with total batch of {}.".format(total_batch))

    for i, sample in enumerate(val_loader):

        real_A = sample['A']
        real_B = sample['B']

        real_A = Variable(real_A).to(device)
        real_B = Variable(real_B).to(device)

        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        fake_ABA = G_B2A(fake_B)
        fake_BAB = G_A2B(fake_A)

        result = torch.cat((real_A, fake_A, fake_BAB, real_B, fake_B, fake_ABA), 0)
        result = ((result.data + 1) / 2).clamp(0, 1)
        save_image(result, './data/results/generated/DiscoGAN_Edges2Shoes_Results_%03d.png' % (i+1), nrow=8, normalize=True)

    make_gifs_test(results_path, "DiscoGAN")


if __name__ == '__main__':
    batch_size = 8
    test(batch_size)