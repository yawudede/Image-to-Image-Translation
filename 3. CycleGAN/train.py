import os
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from config import *
from horse2zebra import get_horse2zebra_loader
from models import Discriminator, Generator
from utils import make_dirs, get_lr_scheduler, set_requires_grad, sample_images, plot_losses, make_gifs_train

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
    train_horse_loader, train_zebra_loader = get_horse2zebra_loader(purpose='train', batch_size=config.batch_size)
    test_horse_loader, test_zebra_loader = get_horse2zebra_loader(purpose='test', batch_size=config.val_batch_size)
    total_batch = min(len(train_horse_loader), len(train_zebra_loader))

    # Prepare Networks #
    D_A = Discriminator()
    D_B = Discriminator()
    G_A2B = Generator()
    G_B2A = Generator()

    networks = [D_A, D_B, G_A2B, G_B2A]

    for network in networks:
        network.to(device)

    # Loss Function #
    criterion_Adversarial = nn.MSELoss()
    criterion_Cycle = nn.L1Loss()
    criterion_Identity = nn.L1Loss()

    # Optimizers #
    D_A_optim = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
    D_B_optim = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(chain(G_A2B.parameters(), G_B2A.parameters()), lr=config.lr, betas=(0.5, 0.999))

    D_A_optim_scheduler = get_lr_scheduler(D_A_optim)
    D_B_optim_scheduler = get_lr_scheduler(D_B_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses_A, D_losses_B, G_losses = [], [], []

    # Training #
    print("Training CycleGAN started with total epoch of {}.".format(config.num_epochs))
    for epoch in range(config.num_epochs):
        for i, (horse, zebra) in enumerate(zip(train_horse_loader, train_zebra_loader)):

            # Data Preparation #
            real_A = horse.to(device)
            real_B = zebra.to(device)

            # Initialize Optimizers #
            G_optim.zero_grad()
            D_A_optim.zero_grad()
            D_B_optim.zero_grad()

            ###################
            # Train Generator #
            ###################

            set_requires_grad([D_A, D_B], requires_grad=False)

            # Adversarial Loss #
            fake_A = G_B2A(real_B)
            prob_fake_A = D_A(fake_A)
            real_labels = torch.ones(prob_fake_A.size()).to(device)
            G_mse_loss_B2A = criterion_Adversarial(prob_fake_A, real_labels)

            fake_B = G_A2B(real_A)
            prob_fake_B = D_B(fake_B)
            real_labels = torch.ones(prob_fake_B.size()).to(device)
            G_mse_loss_A2B = criterion_Adversarial(prob_fake_B, real_labels)

            # Identity Loss #
            identity_A = G_B2A(real_A)
            G_identity_loss_A = config.lambda_identity * criterion_Identity(identity_A, real_A)

            identity_B = G_A2B(real_B)
            G_identity_loss_B = config.lambda_identity * criterion_Identity(identity_B, real_B)

            # Cycle Loss #
            reconstructed_A = G_B2A(fake_B)
            G_cycle_loss_ABA = config.lambda_cycle * criterion_Cycle(reconstructed_A, real_A)

            reconstructed_B = G_A2B(fake_A)
            G_cycle_loss_BAB = config.lambda_cycle * criterion_Cycle(reconstructed_B, real_B)

            # Calculate Total Generator Loss #
            G_loss = G_mse_loss_B2A + G_mse_loss_A2B + G_identity_loss_A + G_identity_loss_B + G_cycle_loss_ABA + G_cycle_loss_BAB

            # Back Propagation and Update #
            G_loss.backward(retain_graph=True)
            G_optim.step()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad([D_A, D_B], requires_grad=True)

            ## Train Discriminator A ##
            # Real Loss #
            prob_real_A = D_A(real_A)
            real_labels = torch.ones(prob_real_A.size()).to(device)
            D_real_loss_A = criterion_Adversarial(prob_real_A, real_labels)

            # Fake Loss #
            prob_fake_A = D_A(fake_A.detach())
            fake_labels = torch.zeros(prob_fake_A.size()).to(device)
            D_fake_loss_A = criterion_Adversarial(prob_fake_A, fake_labels)

            # Calculate Total Discriminator A Loss #
            D_loss_A = config.lambda_identity * (D_real_loss_A + D_fake_loss_A).mean()

            # Back propagation and Update #
            D_loss_A.backward(retain_graph=True)
            D_A_optim.step()

            ## Train Discriminator B ##
            # Real Loss #
            prob_real_B = D_B(real_B)
            real_labels = torch.ones(prob_real_B.size()).to(device)
            loss_real_B = criterion_Adversarial(prob_real_B, real_labels)

            # Fake Loss #
            prob_fake_B = D_B(fake_B.detach())
            fake_labels = torch.zeros(prob_fake_B.size()).to(device)
            loss_fake_B = criterion_Adversarial(prob_fake_B, fake_labels)

            # Calculate Total Discriminator B Loss #
            D_loss_B = config.lambda_identity * (loss_real_B + loss_fake_B).mean()

            # Back propagation and Update #
            D_loss_B.backward(retain_graph=True)
            D_B_optim.step()

            # Add items to Lists #
            D_losses_A.append(D_loss_A.item())
            D_losses_B.append(D_loss_B.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("CycleGAN | Epoch [{}/{}] | Iterations [{}/{}] | D_A Loss {:.4f} | D_B Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch + 1, config.num_epochs, i + 1, total_batch, np.average(D_losses_A), np.average(D_losses_B), np.average(G_losses)))

                # Save Sample Images #
                sample_images(test_horse_loader, test_zebra_loader, G_A2B, G_B2A, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_A_optim_scheduler.step()
        D_B_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch+1) % config.save_every == 0:
            torch.save(G_A2B.state_dict(), os.path.join(config.weights_path, 'CycleGAN_Generator_A2B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(G_B2A.state_dict(), os.path.join(config.weights_path, 'CycleGAN_Generator_B2A_Epoch_{}.pkl'.format(epoch+1)))

    # Make a GIF file #
    make_gifs_train("CycleGAN", config.samples_path)

    # Plot Losses #
    plot_losses(D_losses_A, D_losses_B, G_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()