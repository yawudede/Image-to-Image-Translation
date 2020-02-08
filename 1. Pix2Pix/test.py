import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from models import *
from facades import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(batch_size):

    results_path = './data/results/generated/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    test_loader = get_facades_loader('test', batch_size)
    total_batch = len(test_loader)

    G = Generator().to(device)
    G.load_state_dict(torch.load('./data/results/Pix2Pix_Generator.pkl'))
    G.eval()

    print("Generating Pix2Pix with total batch of {}.".format(total_batch))
    for i, batch in enumerate(test_loader):

        input = Variable(batch['A'].type(torch.FloatTensor).to(device))
        target = Variable(batch['B'].type(torch.FloatTensor).to(device))
        generated = G(input)

        result = torch.cat((target, input, generated), 0)
        result = ((result.data + 1) / 2).clamp(0, 1)
        save_image(result, './data/results/generated/Pix2Pix_Results_%03d.png' % (i+1),
                   nrow=8, normalize=True)

    make_gifs_test(results_path, "Pix2Pix")


if __name__ == '__main__':
    batch_size = 8
    test(batch_size)