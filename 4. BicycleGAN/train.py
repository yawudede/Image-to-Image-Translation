import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from edges2handbags import get_edges2handbags_loader
from models import Discriminator, Encoder, Generator
from utils import make_dirs, get_lr_scheduler, sample_images, set_requires_grad, make_gifs_train, plot_losses

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
    train_loader = get_edges2handbags_loader(purpose='train', batch_size=config.batch_size)
    val_loader = get_edges2handbags_loader(purpose='val', batch_size=config.batch_size)
    total_batch = len(train_loader)

    # Prepare Networks #
    D_cVAE = Discriminator()
    D_cLR = Discriminator()
    E = Encoder(config.z_dim)
    G = Generator(config.z_dim)

    networks = [D_cVAE, D_cLR, E, G]
    for network in networks:
        network.to(device)

    # Loss Function #
    criterion_Recon = nn.L1Loss()
    criterion_Adversarial = nn.MSELoss()

    # Optimizers #
    D_cVAE_optim = torch.optim.Adam(D_cVAE.parameters(), lr=config.lr, betas=(0.5, 0.999))
    D_cLR_optim = torch.optim.Adam(D_cLR.parameters(), lr=config.lr, betas=(0.5, 0.999))
    E_optim = torch.optim.Adam(E.parameters(), lr=config.lr, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))

    D_cVAE_optim_scheduler = get_lr_scheduler(D_cVAE_optim)
    D_cLR_optim_scheduler = get_lr_scheduler(D_cLR_optim)
    E_optim_scheduler = get_lr_scheduler(E_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses, E_G_losses, G_losses = [], [], []

    # Fixed Noise #
    fixed_noise = torch.randn(config.test_size, config.num_images, config.z_dim).to(device)

    # Training #
    print("Training BicycleGAN started total epoch of {}.".format(config.num_epochs))
    for epoch in range(config.num_epochs):
        for i, (sketch, target) in enumerate(train_loader):

            # Data Preparation #
            sketch = sketch.to(device)
            target = target.to(device)

            # Separate Data for D_cVAE-GAN and D_cLR-GAN #
            cVAE_data = {'sketch': sketch[0].unsqueeze(dim=0), 'target': target[0].unsqueeze(dim=0)}
            cLR_data = {'sketch': sketch[1].unsqueeze(dim=0), 'target': target[1].unsqueeze(dim=0)}

            # Initialize Optimizers #
            D_cVAE_optim.zero_grad()
            D_cLR_optim.zero_grad()
            E_optim.zero_grad()
            G_optim.zero_grad()

            # Train Discriminators #
            set_requires_grad([D_cVAE, D_cLR], requires_grad=True)

            ################################
            # Train Discriminator cVAE-GAN #
            ################################

            # Initialize Optimizers #
            D_cVAE_optim.zero_grad()
            D_cLR_optim.zero_grad()
            E_optim.zero_grad()
            G_optim.zero_grad()

            # Encode Latent Vector #
            mean, std = E(cVAE_data['target'])
            random_z = torch.randn(1, config.z_dim).to(device)
            encoded_z = mean + (random_z * std)

            # Generate Fake Image #
            fake_image_cVAE = G(cVAE_data['sketch'], encoded_z)

            # Forward to Discriminator cVAE-GAN #
            prob_real_D_cVAE_1, prob_real_D_cVAE_2 = D_cVAE(cVAE_data['target'])
            prob_fake_D_cVAE_1, prob_fake_D_cVAE_2 = D_cVAE(fake_image_cVAE.detach())

            # Adversarial Loss using cVAE_1 #
            real_labels = torch.ones(prob_real_D_cVAE_1.size()).to(device)
            D_cVAE_1_real_loss = criterion_Adversarial(prob_real_D_cVAE_1, real_labels)

            fake_labels = torch.zeros(prob_fake_D_cVAE_1.size()).to(device)
            D_cVAE_1_fake_loss = criterion_Adversarial(prob_fake_D_cVAE_1, fake_labels)

            D_cVAE_1_loss = D_cVAE_1_real_loss + D_cVAE_1_fake_loss

            # Adversarial Loss using cVAE_2 #
            real_labels = torch.ones(prob_real_D_cVAE_2.size()).to(device)
            D_cVAE_2_real_loss = criterion_Adversarial(prob_real_D_cVAE_2, real_labels)

            fake_labels = torch.zeros(prob_fake_D_cVAE_2.size()).to(device)
            D_cVAE_2_fake_loss = criterion_Adversarial(prob_fake_D_cVAE_2, fake_labels)

            D_cVAE_2_loss = D_cVAE_2_real_loss + D_cVAE_2_fake_loss

            ###########################
            # Train Discriminator cLR #
            ###########################

            # Initialize Optimizers #
            D_cVAE_optim.zero_grad()
            D_cLR_optim.zero_grad()
            E_optim.zero_grad()
            G_optim.zero_grad()

            # Generate Fake Image using Random Latent Vector #
            random_z = torch.randn(1, config.z_dim).to(device)
            fake_image_cLR = G(cLR_data['sketch'], random_z)

            # Forward to Discriminator cLR-GAN #
            prob_real_D_cLR_1, prob_real_D_cLR_2 = D_cLR(cLR_data['target'])
            prob_fake_D_cLR_1, prob_fake_D_cLR_2 = D_cLR(fake_image_cLR.detach())

            # Adversarial Loss using cLR-1 #
            real_labels = torch.ones(prob_real_D_cLR_1.size()).to(device)
            D_cLR_1_real_loss = criterion_Adversarial(prob_real_D_cLR_1, real_labels)

            fake_labels = torch.zeros(prob_fake_D_cLR_1.size()).to(device)
            D_cLR_1_fake_loss = criterion_Adversarial(prob_fake_D_cLR_1, fake_labels)

            D_cLR_1_loss = D_cLR_1_real_loss + D_cLR_1_fake_loss

            # Adversarial Loss using cLR-2 #
            real_labels = torch.ones(prob_real_D_cLR_2.size()).to(device)
            D_cLR_2_real_loss = criterion_Adversarial(prob_real_D_cLR_2, real_labels)

            fake_labels = torch.zeros(prob_fake_D_cLR_2.size()).to(device)
            D_cLR_2_fake_loss = criterion_Adversarial(prob_fake_D_cLR_2, fake_labels)

            D_cLR_2_loss = D_cLR_2_real_loss + D_cLR_2_fake_loss

            # Calculate Total Discriminator Loss #
            D_loss = D_cVAE_1_loss + D_cVAE_2_loss + D_cLR_1_loss + D_cLR_2_loss

            # Back Propagation and Update #
            D_loss.backward()
            D_cVAE_optim.step()
            D_cLR_optim.step()

            set_requires_grad([D_cVAE, D_cLR], requires_grad=False)

            ###############################
            # Train Encoder and Generator #
            ###############################

            # Initialize Optimizers #
            D_cVAE_optim.zero_grad()
            D_cLR_optim.zero_grad()
            E_optim.zero_grad()
            G_optim.zero_grad()

            # Encode Latent Vector #
            mean, std = E(cVAE_data['target'])
            random_z = torch.randn(1, config.z_dim).to(device)
            encoded_z = mean + (random_z * std)

            # Generate Fake Image #
            fake_image_cVAE = G(cVAE_data['sketch'], encoded_z)
            prob_fake_D_cVAE_1, prob_fake_D_cVAE_2 = D_cVAE(fake_image_cVAE)

            # Adversarial Loss using cVAE #
            real_labels = torch.ones(prob_fake_D_cVAE_1.size()).to(device)
            E_G_adv_cVAE_1_loss = criterion_Adversarial(prob_fake_D_cVAE_1, real_labels)

            real_labels = torch.ones(prob_fake_D_cVAE_2.size()).to(device)
            E_G_adv_cVAE_2_loss = criterion_Adversarial(prob_fake_D_cVAE_2, real_labels)

            E_G_adv_cVAE_loss = E_G_adv_cVAE_1_loss + E_G_adv_cVAE_2_loss

            # Generate Fake Image using Random Latent Vector #
            random_z = torch.randn(1, config.z_dim).to(device)
            fake_image_cLR = G(cLR_data['sketch'], random_z)
            prob_fake_D_cLR_1, prob_fake_D_cLR_2 = D_cLR(fake_image_cLR)

            # Adversarial Loss of cLR #
            real_labels = torch.ones(prob_fake_D_cLR_1.size()).to(device)
            E_G_adv_cLR_1_loss = criterion_Adversarial(prob_fake_D_cLR_1, real_labels)

            real_labels = torch.ones(prob_fake_D_cLR_2.size()).to(device)
            E_G_adv_cLR_2_loss = criterion_Adversarial(prob_fake_D_cLR_2, real_labels)

            E_G_adv_cLR_loss = E_G_adv_cLR_1_loss + E_G_adv_cLR_2_loss

            # KL Divergence with N ~ (0, 1) #
            E_KL_div_loss = config.lambda_KL * torch.sum(0.5 * (mean ** 2 + std - 2 * torch.log(std) - 1))

            # Reconstruction Loss #
            E_G_recon_loss = config.lambda_Image * criterion_Recon(fake_image_cVAE, cVAE_data['target'])

            # Total Encoder and Generator Loss ##
            E_G_loss = E_G_adv_cVAE_loss + E_G_adv_cLR_loss + E_KL_div_loss + E_G_recon_loss

            # Back Propagation and Update #
            E_G_loss.backward()
            E_optim.step()
            G_optim.step()

            ########################
            # Train Generator Only #
            ########################

            # Initialize Optimizers #
            D_cVAE_optim.zero_grad()
            D_cLR_optim.zero_grad()
            E_optim.zero_grad()
            G_optim.zero_grad()

            # Generate Fake Image using Random Latent Vector #
            random_z = torch.randn(1, config.z_dim).to(device)
            fake_image_cLR = G(cLR_data['sketch'], random_z)
            mean, std = E(fake_image_cLR)

            # Reconstruction Loss #
            G_recon_loss = criterion_Recon(mean, random_z)

            # Calculate Total Generator Loss #
            G_loss = config.lambda_Z * G_recon_loss

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            E_G_losses.append(E_G_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("BicycleGAN | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | E_G Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(D_losses), np.average(E_G_losses), np.average(G_losses)))

                # Save Sample Images #
                sample_images(val_loader, G, fixed_noise, epoch, config.num_images, config.samples_path)


        # Adjust Learning Rate #
        D_cVAE_optim_scheduler.step()
        D_cLR_optim_scheduler.step()
        E_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Models #
        if (epoch+1) % config.save_every == 0:
            torch.save(G.state_dict(), os.path.join(config.weights_path, 'BicycleGAN_Generator_Epoch_{}.pkl'.format(epoch+1)))

    # Make a GIF file #
    make_gifs_train("BicycleGAN", config.samples_path)

    # Plot Losses #
    plot_losses(D_losses, E_G_losses, G_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()