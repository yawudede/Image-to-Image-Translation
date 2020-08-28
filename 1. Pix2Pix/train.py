import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from facades import *
from models import Discriminator, Generator
from utils import make_dirs, set_requires_grad, get_lr_scheduler, sample_images, plot_losses, make_gifs_train

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():

    # Fix Seed for Reproducibility #
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Samples, Weights and Results Path #
    paths = [config.samples_path, config.weights_path, config.plots_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    train_loader = get_facades_loader('train', config.batch_size)
    val_loader = get_facades_loader('val', config.val_batch_size)
    total_batch = len(train_loader)

    # Prepare Networks #
    D = Discriminator().to(device)
    G = Generator().to(device)

    # Criterion #
    criterion_Adversarial = nn.BCELoss()
    criterion_Pixelwise = nn.L1Loss()

    # Optimizers #
    D_optim = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses, G_losses = [], []

    # Training #
    print("Training Pix2Pix started with total epoch of {}.".format(config.num_epochs))

    for epoch in range(config.num_epochs):

        for i, (real_A, real_B) in enumerate(train_loader):

            # Data Preparation #
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            ###################
            # Train Generator #
            ###################

            # Prevent Discriminator Update during Generator Update #
            set_requires_grad(D, requires_grad=False)

            # Adversarial Loss #
            fake_B = G(real_A)
            prob_fake = D(fake_B, real_A)
            real_labels = torch.ones(prob_fake.size()).to(device)
            G_loss_fake = criterion_Adversarial(prob_fake, real_labels)

            # Pixel-Wise Loss #
            G_loss_pixelwise = criterion_Pixelwise(fake_B, real_B)

            # Calculate Total Generator Loss #
            G_loss = G_loss_fake + config.l1_lambda * G_loss_pixelwise

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            #######################
            # Train Discriminator #
            #######################

            # Prevent Discriminator Update during Generator Update #
            set_requires_grad(D, requires_grad=True)

            # Adversarial Loss #
            prob_real = D(real_B, real_A)
            real_labels = torch.ones(prob_real.size()).to(device)
            D_real_loss = criterion_Adversarial(prob_real, real_labels)

            fake_B = G(real_A)
            prob_fake = D(fake_B.detach(), real_A)
            fake_labels = torch.zeros(prob_fake.size()).to(device)
            D_fake_loss = criterion_Adversarial(prob_fake, fake_labels)

            # Calculate Total Discriminator Loss #
            D_loss = torch.mean(D_real_loss + D_fake_loss)

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("Pix2Pix | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                sample_images(val_loader, G, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch + 1) % config.save_every == 0:
            torch.save(G.state_dict(), os.path.join(config.weights_path, 'Pix2Pix_Generator_Epoch_{}.pkl'.format(epoch+1)))

    # Make a GIF file #
    make_gifs_train('Pix2Pix', config.samples_path)

    # Plot Losses #
    plot_losses(D_losses, G_losses, config.plots_path)

    print("Training finished.")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()