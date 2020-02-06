import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from itertools import chain

from models import *
from horse2zebra import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(batch_size):

    results_path = './data/results/generated/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    testloader_horse, testloader_zebra = get_horse2zebra_loader(1, 'test')
    total_batch = min(len(testloader_horse), len(testloader_zebra))

    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    G_A2B.load_state_dict(torch.load('./data/results/CycleGAN_Generator_A2B.pkl'))
    G_B2A.load_state_dict(torch.load('./data/results/CycleGAN_Generator_B2A.pkl'))

    G_A2B.eval()
    G_B2A.eval()

    print("Generating CycleGAN with total batch of {}.".format(total_batch))

    for i, (horse, zebra) in enumerate(zip(testloader_horse, testloader_zebra)):

        real_A = Variable(horse).to(device)
        real_B = Variable(zebra).to(device)

        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        fake_ABA = G_B2A(fake_B)
        fake_BAB = G_A2B(fake_A)

        result = torch.cat((real_A, fake_B, fake_ABA, real_B, fake_A, fake_BAB), 0)
        result = ((result.data + 1) / 2).clamp(0, 1)
        save_image(result, './data/results/generated/CycleGAN_Horse2Zebra_Results_%03d.png' % (i+1),
                   nrow=6, normalize=True)

    make_gifs_test(results_path, "CycleGAN")


if __name__ == '__main__':
    batch_size = 2
    test(batch_size)