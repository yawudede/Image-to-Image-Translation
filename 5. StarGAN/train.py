import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from celeba import get_celeba_loader
from models import Discriminator, Generator
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
    train_loader = get_celeba_loader('train', config.batch_size, config.selected_attrs)
    total_batch = len(train_loader)

    fixed_image, original_label = next(iter(train_loader))
    fixed_image = fixed_image.to(device)
    fixed_labels_list = create_labels(original_label, config.selected_attrs)

    # Prepare Networks #
    D = Discriminator(num_classes=len(config.selected_attrs)).to(device)
    G = Generator(num_classes=len(config.selected_attrs)).to(device)

    # Optimizers #
    D_optim = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses, G_losses = [], []

    # Train #
    print("Training StarGAN started with total epoch of {}.".format(config.num_epochs))

    for epoch in range(config.num_epochs):

        for i, batch in enumerate(train_loader):

            # Data Preparation #
            real_image, label = next(iter(train_loader))

            real_image = real_image.to(device)
            label = label.to(device)

            rand_idx = torch.randperm(label.size(0))
            target_label = label[rand_idx].to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad(D, requires_grad=True)

            # Discriminiator Loss using Real Image #
            prob_real_src, prob_real_cls = D(real_image)
            D_real_loss = - torch.mean(prob_real_src)
            D_cls_loss = config.lambda_cls * criterion_CLS(prob_real_cls, label)

            # Discriminiator Loss using Generated Image #
            fake_image = G(real_image, target_label)
            prob_fake_src, prob_fake_cls = D(fake_image.detach())
            D_fake_loss = torch.mean(prob_fake_src)

            # Discriminiator Loss using Wasserstein GAN Gradient Penalty #
            D_gp_loss = config.lambda_gp * get_gradient_penalty(real_image, fake_image, D)

            # Calculate Total Discriminator Loss #
            D_loss = D_real_loss + D_fake_loss + D_cls_loss + D_gp_loss

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())

            ###################
            # Train Generator #
            ###################

            if (i+1) % config.n_critics == 0:

                # Prevent Discriminator Update during Generator Update #
                set_requires_grad(D, requires_grad=False)

                # Initialize Optimizers #
                D_optim.zero_grad()
                G_optim.zero_grad()

                # Generator Loss using Fake Images #
                fake_image = G(real_image, target_label)
                prob_fake_src, prob_fake_cls = D(fake_image)
                G_fake_loss = -torch.mean(prob_fake_src)
                G_cls_loss = config.lambda_cls * criterion_CLS(prob_fake_cls, target_label)

                # Reconstruction Loss #
                recon_image = G(fake_image, label)
                G_recon_loss = config.lambda_recon * torch.mean(torch.abs(real_image - recon_image))

                # Calculate Total Generator Loss #
                G_loss = G_fake_loss + G_recon_loss + G_cls_loss

                # Back Propagation and Update #
                G_loss.backward()
                G_optim.step()

                # Add items to Lists #
                G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("StarGAN | Epoch [{}/{}] | Iteration [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                save_samples(fixed_image, fixed_labels_list, G, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch+1) % config.save_every == 0:
            torch.save(G.state_dict(), os.path.join(config.weights_path, 'StarGAN_Generator_Epoch_{}.pkl'.format(epoch+1)))

    # Make a GIF file #
    make_gifs_train('StarGAN', config.samples_path)

    # Plot Losses #
    plot_losses(D_losses, G_losses, config.num_epochs, config.plots_path)

    print("Training Finished.")


if __name__ == "__main__":
    train()