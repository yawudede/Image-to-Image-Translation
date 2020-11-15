import os
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from selfie2anime import get_selfie2anime_loader
from models import Discriminator, Generator
from utils import save_samples, RhoClipper, get_lr_scheduler, set_requires_grad, plot_losses, make_gifs_train, make_dirs

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

    # Samples, Weights, and Plots Path #
    paths = [config.samples_path, config.weights_path, config.plots_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    train_loader_selfie, train_loader_anime = get_selfie2anime_loader('train', config.batch_size)
    total_batch = max(len(train_loader_selfie), len(train_loader_anime))

    test_loader_selfie, test_loader_anime = get_selfie2anime_loader('test', config.val_batch_size)

    # Prepare Networks #
    D_A = Discriminator(num_layers=7)
    D_B = Discriminator(num_layers=7)
    L_A = Discriminator(num_layers=5)
    L_B = Discriminator(num_layers=5)
    G_A2B = Generator(image_size=config.crop_size, num_blocks=config.num_blocks)
    G_B2A = Generator(image_size=config.crop_size, num_blocks=config.num_blocks)

    networks = [D_A, D_B, L_A, L_B, G_A2B, G_B2A]

    for network in networks:
        network.to(device)

    # Loss Function #
    Adversarial_loss = nn.MSELoss()
    Cycle_loss = nn.L1Loss()
    BCE_loss = nn.BCEWithLogitsLoss()

    # Optimizers #
    D_optim = torch.optim.Adam(chain(D_A.parameters(), D_B.parameters(), L_A.parameters(), L_B.parameters()),
                               lr=config.lr, betas=(0.5, 0.999), weight_decay=0.0001)
    G_optim = torch.optim.Adam(chain(G_A2B.parameters(), G_B2A.parameters()),
                               lr=config.lr, betas=(0.5, 0.999), weight_decay=0.0001)

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Rho Clipper to constraint the value of rho in AdaILN and ILN #
    Rho_Clipper = RhoClipper(0, 1)

    # Lists #
    D_losses = []
    G_losses = []

    # Train #
    print("Training U-GAT-IT started with total epoch of {}.".format(config.num_epochs))
    for epoch in range(config.num_epochs):

        for i, (selfie, anime) in enumerate(zip(train_loader_selfie, train_loader_anime)):

            # Data Preparation #
            real_A = selfie.to(device)
            real_B = anime.to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad([D_A, D_B, L_A, L_B], requires_grad=True)

            # Forward Data #
            fake_B, _, _ = G_A2B(real_A)
            fake_A, _, _ = G_B2A(real_B)

            G_real_A, G_real_A_cam, _ = D_A(real_A)
            L_real_A, L_real_A_cam, _ = L_A(real_A)
            G_real_B, G_real_B_cam, _ = D_B(real_B)
            L_real_B, L_real_B_cam, _ = L_B(real_B)

            G_fake_A, G_fake_A_cam, _ = D_A(fake_A)
            L_fake_A, L_fake_A_cam, _ = L_A(fake_A)
            G_fake_B, G_fake_B_cam, _ = D_B(fake_B)
            L_fake_B, L_fake_B_cam, _ = L_B(fake_B)

            # Adversarial Loss of Discriminator #
            real_labels = torch.ones(G_real_A.shape).to(device)
            D_ad_real_loss_GA = Adversarial_loss(G_real_A, real_labels)

            fake_labels = torch.zeros(G_fake_A.shape).to(device)
            D_ad_fake_loss_GA = Adversarial_loss(G_fake_A, fake_labels)

            D_ad_loss_GA = D_ad_real_loss_GA + D_ad_fake_loss_GA

            real_labels = torch.ones(G_real_A_cam.shape).to(device)
            D_ad_cam_real_loss_GA = Adversarial_loss(G_real_A_cam, real_labels)

            fake_labels = torch.zeros(G_fake_A_cam.shape).to(device)
            D_ad_cam_fake_loss_GA = Adversarial_loss(G_fake_A_cam, fake_labels)

            D_ad_cam_loss_GA = D_ad_cam_real_loss_GA + D_ad_cam_fake_loss_GA

            real_labels = torch.ones(G_real_B.shape).to(device)
            D_ad_real_loss_GB = Adversarial_loss(G_real_B, real_labels)

            fake_labels = torch.zeros(G_fake_B.shape).to(device)
            D_ad_fake_loss_GB = Adversarial_loss(G_fake_B, fake_labels)

            D_ad_loss_GB = D_ad_real_loss_GB + D_ad_fake_loss_GB

            real_labels = torch.ones(G_real_B_cam.shape).to(device)
            D_ad_cam_real_loss_GB = Adversarial_loss(G_real_B_cam, real_labels)

            fake_labels = torch.zeros(G_fake_B_cam.shape).to(device)
            D_ad_cam_fake_loss_GB = Adversarial_loss(G_fake_B_cam, fake_labels)

            D_ad_cam_loss_GB = D_ad_cam_real_loss_GB + D_ad_cam_fake_loss_GB

            # Adversarial Loss of L #
            real_labels = torch.ones(L_real_A.shape).to(device)
            D_ad_real_loss_LA = Adversarial_loss(L_real_A, real_labels)

            fake_labels = torch.zeros(L_fake_A.shape).to(device)
            D_ad_fake_loss_LA = Adversarial_loss(L_fake_A, fake_labels)

            D_ad_loss_LA = D_ad_real_loss_LA + D_ad_fake_loss_LA

            real_labels = torch.ones(L_real_A_cam.shape).to(device)
            D_ad_cam_real_loss_LA = Adversarial_loss(L_real_A_cam, real_labels)

            fake_labels = torch.zeros(L_fake_A_cam.shape).to(device)
            D_ad_cam_fake_loss_LA = Adversarial_loss(L_fake_A_cam, fake_labels)

            D_ad_cam_loss_LA = D_ad_cam_real_loss_LA + D_ad_cam_fake_loss_LA

            real_labels = torch.ones(L_real_B.shape).to(device)
            D_ad_real_loss_LB = Adversarial_loss(L_real_B, real_labels)

            fake_labels = torch.zeros(L_fake_B.shape).to(device)
            D_ad_fake_loss_LB = Adversarial_loss(L_fake_B, fake_labels)

            D_ad_loss_LB = D_ad_real_loss_LB + D_ad_fake_loss_LB

            real_labels = torch.ones(L_real_B_cam.shape).to(device)
            D_ad_cam_real_loss_LB = Adversarial_loss(L_real_B_cam, real_labels)

            fake_labels = torch.zeros(L_fake_B_cam.shape).to(device)
            D_ad_cam_fake_loss_LB = Adversarial_loss(L_fake_B_cam, fake_labels)

            D_ad_cam_loss_LB = D_ad_cam_real_loss_LB + D_ad_cam_fake_loss_LB

            # Calculate Each Discriminator Loss #
            D_loss_A = D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA
            D_loss_B = D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB

            # Calculate Total Discriminator Loss #
            D_loss = D_loss_A + D_loss_B

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            ###################
            # Train Generator #
            ###################

            set_requires_grad([D_A, D_B, L_A, L_B], requires_grad=False)

            # Forward Data #
            fake_B, fake_B_cam, _ = G_A2B(real_A)
            fake_A, fake_A_cam, _ = G_B2A(real_B)

            fake_ABA, _, _ = G_B2A(fake_B)
            fake_BAB, _, _ = G_A2B(fake_A)

            fake_A2A, fake_A2A_cam, _ = G_A2B(real_A)
            fake_B2B, fake_B2B_cam, _ = G_B2A(real_B)

            G_fake_A, G_fake_A_cam, _ = D_A(fake_A)
            L_fake_A, L_fake_A_cam, _ = L_A(fake_A)
            G_fake_B, G_fake_B_cam, _ = D_B(fake_B)
            L_fake_B, L_fake_B_cam, _ = L_B(fake_B)

            # Adversarial Loss of Generator #
            real_labels = torch.ones(G_fake_A.shape).to(device)
            G_adv_fake_loss_A = Adversarial_loss(G_fake_A, real_labels)

            real_labels = torch.ones(G_fake_A_cam.shape).to(device)
            G_adv_cam_fake_loss_A = Adversarial_loss(G_fake_A_cam, real_labels)

            G_adv_loss_A = G_adv_fake_loss_A + G_adv_cam_fake_loss_A

            real_labels = torch.ones(G_fake_B.shape).to(device)
            G_adv_fake_loss_B = Adversarial_loss(G_fake_B, real_labels)

            real_labels = torch.ones(G_fake_B_cam.shape).to(device)
            G_adv_cam_fake_loss_B = Adversarial_loss(G_fake_B_cam, real_labels)

            G_adv_loss_B = G_adv_fake_loss_B + G_adv_cam_fake_loss_B

            # Adversarial Loss of L #
            real_labels = torch.ones(L_fake_A.shape).to(device)
            L_adv_fake_loss_A = Adversarial_loss(L_fake_A, real_labels)

            real_labels = torch.ones(L_fake_A_cam.shape).to(device)
            L_adv_cam_fake_loss_A = Adversarial_loss(L_fake_A_cam, real_labels)

            L_adv_loss_A = L_adv_fake_loss_A + L_adv_cam_fake_loss_A

            real_labels = torch.ones(L_fake_B.shape).to(device)
            L_adv_fake_loss_B = Adversarial_loss(L_fake_B, real_labels)

            real_labels = torch.ones(L_fake_B_cam.shape).to(device)
            L_adv_cam_fake_loss_B = Adversarial_loss(L_fake_B_cam, real_labels)

            L_adv_loss_B = L_adv_fake_loss_B + L_adv_cam_fake_loss_B

            # Cycle Consistency Loss #
            G_recon_loss_A = Cycle_loss(fake_ABA, real_A)
            G_recon_loss_B = Cycle_loss(fake_BAB, real_B)

            G_identity_loss_A = Cycle_loss(fake_A2A, real_A)
            G_identity_loss_B = Cycle_loss(fake_B2B, real_B)

            G_cycle_loss_A = G_recon_loss_A + G_identity_loss_A
            G_cycle_loss_B = G_recon_loss_B + G_identity_loss_B

            # CAM Loss #
            real_labels = torch.ones(fake_A_cam.shape).to(device)
            G_cam_real_loss_A = BCE_loss(fake_A_cam, real_labels)

            fake_labels = torch.zeros(fake_A2A_cam.shape).to(device)
            G_cam_fake_loss_A = BCE_loss(fake_A2A_cam, fake_labels)

            G_cam_loss_A = G_cam_real_loss_A + G_cam_fake_loss_A

            real_labels = torch.ones(fake_B_cam.shape).to(device)
            G_cam_real_loss_B = BCE_loss(fake_B_cam, real_labels)

            fake_labels = torch.zeros(fake_B2B_cam.shape).to(device)
            G_cam_fake_loss_B = BCE_loss(fake_B2B_cam, fake_labels)

            G_cam_loss_B = G_cam_real_loss_B + G_cam_fake_loss_B

            # Calculate Each Generator Loss #
            G_loss_A = G_adv_loss_A + L_adv_loss_A + config.lambda_cycle * G_cycle_loss_A + config.lambda_cam * G_cam_loss_A
            G_loss_B = G_adv_loss_B + L_adv_loss_B + config.lambda_cycle * G_cycle_loss_B + config.lambda_cam * G_cam_loss_B

            # Calculate Total Generator Loss #
            G_loss = G_loss_A + G_loss_B

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Apply Rho Clipper to Generators #
            G_A2B.apply(Rho_Clipper)
            G_B2A.apply(Rho_Clipper)

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("U-GAT-IT | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, config.num_epochs, i+1, total_batch, np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                save_samples(test_loader_selfie, G_A2B, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch+1) % config.save_every == 0:
            torch.save(D_A.state_dict(), os.path.join(config.weights_path, 'U-GAT-IT_D_A_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(D_B.state_dict(), os.path.join(config.weights_path, 'U-GAT-IT_D_B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(L_A.state_dict(), os.path.join(config.weights_path, 'U-GAT-IT_L_A_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(L_B.state_dict(), os.path.join(config.weights_path, 'U-GAT-IT_L_B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(G_A2B.state_dict(), os.path.join(config.weights_path, 'U-GAT-IT_G_A2B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(G_B2A.state_dict(), os.path.join(config.weights_path, 'U-GAT-IT_G_B2A_Epoch_{}.pkl'.format(epoch+1)))

    # Plot Losses #
    plot_losses(D_losses, G_losses, config.num_epochs, config.plots_path)

    # Make a GIF file #
    make_gifs_train('U-GAT-IT', config.samples_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()