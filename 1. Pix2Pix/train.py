import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np

from facades import *
from models import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(batch_size, num_epochs):

    # Results Path #
    results_path = './data/results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Prepare Data Loader #
    train_loader = get_facades_loader('train', batch_size)
    val_loader = get_facades_loader('val', 3)
    total_batch = len(train_loader)

    # Networks #
    D = Discriminator().to(device)
    G = Generator().to(device)

    D.apply(weights_init)
    G.apply(weights_init)

    patch = (1, 30, 30)

    # Criterion #
    criterion_GAN = nn.BCELoss()
    criterion_Pixelwise = nn.L1Loss()
    LAMBDA = 100

    D_losses, G_losses = [], []

    # Optimizers #
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training #
    print("Training Pix2Pix started with batch size of {} and total batch of {}.".format(batch_size, total_batch))
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):

            # Data Preparation #
            real_A = Variable(batch['A'].type(torch.FloatTensor).to(device))
            real_B = Variable(batch['B'].type(torch.FloatTensor).to(device))

            # Labels #
            valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))).to(device))
            fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))).to(device))

            ### Train Generator ###
            # Initialize #
            optim_G.zero_grad()

            # Adversarial Loss #
            fake_B = G(real_A)
            prob_fake = D(fake_B, real_A)
            G_loss_fake = criterion_GAN(prob_fake, valid)

            # Pixel-wise Loss #
            G_loss_pixelwise = criterion_Pixelwise(fake_B, real_B)

            # Total Generator Loss #
            G_loss = G_loss_fake + LAMBDA * G_loss_pixelwise

            # Back Propagation and Update #
            G_loss.backward(retain_graph=True)
            optim_G.step()

            ### Train Discriminator ###
            # Initialize #
            optim_D.zero_grad()

            # Adversarial Loss #
            prob_real = D(real_B, real_A)
            D_loss_real = criterion_GAN(prob_real, valid)

            prob_fake = D(fake_B.detach(), real_A)
            D_loss_fake = criterion_GAN(prob_fake, fake)

            # Total Discriminator Loss #
            D_loss = D_loss_real + D_loss_fake

            # Back Propagation and Update #
            D_loss.backward(retain_graph=True)
            optim_D.zero_grad()

            ### Print Statistics ###
            if (i+1) % 100 == 0:
                print("Pix2Pix | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}".
                      format(epoch+1, num_epochs, i+1, total_batch, D_loss.item(), G_loss.item()))

                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())

    # Save Images #
        sample_images(val_loader, epoch, G, results_path)

    make_gifs_train('Pix2Pix', results_path)
    plot_losses(D_losses, G_losses, num_epochs, results_path)

    # Save Models #
    torch.save(D.state_dict(), './data/results/Pix2Pix_Discriminator.pkl')
    torch.save(G.state_dict(), './data/results/Pix2Pix_Generator.pkl')


if __name__ == '__main__':
    torch.cuda.empty_cache()

    batch_size = 1
    num_epochs = 60
    train(batch_size, num_epochs)