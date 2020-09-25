import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os
import numpy as np
import time

from config import *
from edges2shoes import get_edges2shoes_loader
from models import AdaIN_Generator, Multi_Discriminator
from utils import *

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
    train_loader_A, train_loader_B = get_edges2shoes_loader('train', config.batch_size)
    test_loader_A, test_loader_B = get_edges2shoes_loader('test', config.val_batch_size)
    total_batch = min(len(train_loader_A), len(train_loader_B))

    # Prepare Networks #
    D_A = Multi_Discriminator()
    D_B = Multi_Discriminator()
    G_A2B = AdaIN_Generator()
    G_B2A = AdaIN_Generator()

    networks = [D_A, D_B, G_A2B, G_B2A]
    for network in networks:
        network.to(device)

    # Loss Function #
    criterion_Recon = nn.L1Loss()

    # Optimizers #
    D_params = list(D_A.parameters()) + list(D_B.parameters())
    G_params = list(G_A2B.parameters()) + list(G_B2A.parameters())

    D_optim = torch.optim.Adam([p for p in D_params if p.requires_grad],
                               lr=config.lr, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam([p for p in G_params if p.requires_grad],
                               lr=config.lr, betas=(0.5, 0.999))

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses, G_losses = [], []

    # Fixed Style #
    fixed_style_A = torch.randn(config.batch_size, config.style_dim, 1, 1).to(device)
    fixed_style_B = torch.randn(config.batch_size, config.style_dim, 1, 1).to(device)

    # Train #
    print("Training MUNIT started with total epoch of {}.".format(config.num_epochs))
    for epoch in range(config.num_epochs):
        for i, (real_A, real_B) in enumerate(zip(train_loader_A, train_loader_B)):

            G_A2B.train()
            G_B2A.train()

            # Data Preparation #
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad([D_A, D_B], requires_grad=True)

            # Define Style #
            style_a = torch.randn(real_A.size(0), config.style_dim, 1, 1).to(device)
            style_b = torch.randn(real_B.size(0), config.style_dim, 1, 1).to(device)

            # Extract Content from Real Images #
            content_A, _ = G_B2A.encode(real_A)
            content_B, _ = G_A2B.encode(real_B)

            # Generate Fake Images #
            fake_A = G_B2A.decode(content_B, style_a)
            fake_B = G_A2B.decode(content_A, style_b)

            prob_reals_A = D_A(real_A)
            prob_reals_B = D_B(real_B)

            D_loss_real_A = criterion_Adversarial(prob_reals_A, labels='real')
            D_loss_real_B = criterion_Adversarial(prob_reals_B, labels='real')

            # Calculate Discriminator Loss using Real Images #
            D_loss_real = D_loss_real_A + D_loss_real_B

            prob_fakes_A = D_A(fake_A.detach())
            prob_fakes_B = D_B(fake_B.detach())

            D_loss_fake_A = criterion_Adversarial(prob_fakes_A, labels='fake')
            D_loss_fake_B = criterion_Adversarial(prob_fakes_B, labels='fake')

            # Calculate Discriminator Loss using Fake Images #
            D_loss_fake = D_loss_fake_A + D_loss_fake_B

            # Calculate Total Discriminator Loss #
            D_loss = D_loss_real + D_loss_fake

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            ###################
            # Train Generator #
            ###################

            set_requires_grad([D_A, D_B], requires_grad=False)

            # Define Style #
            style_A = torch.randn(real_A.size(0), config.style_dim, 1, 1).to(device)
            style_B = torch.randn(real_B.size(0), config.style_dim, 1, 1).to(device)

            # Encode Content and Style #
            content_A, style_A_prime = G_B2A.encode(real_A)
            content_B, style_B_prime = G_A2B.encode(real_B)

            # Decode Content and Style Within the Domain #
            fake_A_recon = G_B2A.decode(content_A, style_A_prime)
            fake_B_recon = G_A2B.decode(content_B, style_B_prime)

            # Decode Content and Style Across Domain#
            fake_A = G_B2A.decode(content_B, style_A)
            fake_B = G_A2B.decode(content_A, style_B)

            # Adversarial Loss using Fake A #
            prob_fakes_A = D_A(fake_A)
            G_loss_fake_A = criterion_Adversarial(prob_fakes_A, labels='real')

            # Adversarial Loss using Fake B #
            prob_fakes_B = D_B(fake_B)
            G_loss_fake_B = criterion_Adversarial(prob_fakes_B, labels='real')

            # Encode Content and Style using Fake Images #
            content_B_recon, style_A_recon = G_B2A.encode(fake_A)
            content_A_recon, style_B_recon = G_A2B.encode(fake_B)

            # Style Reconstruction #
            G_loss_recon_style_A = criterion_Recon(style_A_recon, style_A)
            G_loss_recon_style_B = criterion_Recon(style_B_recon, style_B)

            # Content Reconstruction #
            G_loss_recon_content_A = criterion_Recon(content_A_recon, content_A)
            G_loss_recon_content_B = criterion_Recon(content_B_recon, content_B)

            # GT Reconstruction #
            G_loss_recon_A = criterion_Recon(fake_A_recon, real_A)
            G_loss_recon_B = criterion_Recon(fake_B_recon, real_B)

            # Calculate Total Generator Loss #
            G_loss = G_loss_fake_A + G_loss_fake_B + \
                     config.lambda_recon * G_loss_recon_A + config.lambda_recon * G_loss_recon_B + \
                     G_loss_recon_content_A + G_loss_recon_content_B + \
                     G_loss_recon_style_A + G_loss_recon_style_B

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i + 1) % config.print_every == 0:
                print("MUNIT | Epoch [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch + 1, config.num_epochs, i + 1, total_batch, np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                sample_images(test_loader_A, test_loader_B, G_A2B, G_B2A, fixed_style_A, fixed_style_B, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch + 1) % config.save_every == 0:
            torch.save(G_A2B.state_dict(), os.path.join(config.weights_path, 'MUNIT_Generator_A2B_Epoch_{}.pkl'.format(epoch + 1)))
            torch.save(G_B2A.state_dict(), os.path.join(config.weights_path, 'MUNIT_Generator_B2A_Epoch_{}.pkl'.format(epoch + 1)))

    # Make a GIF file #
    make_gifs_train("MUNIT", config.samples_path)

    # Plot Losses #
    plot_losses(D_losses, G_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()