import os
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from horse2zebra import get_horse2zebra_loader
from models import Attention, Discriminator, Generator
from image_pool import ImagePool, ImageMaskPool
from utils import make_dirs, get_lr_scheduler, set_requires_grad, save_samples, plot_losses, make_gifs_train


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
    train_horse_loader, train_zebra_loader = get_horse2zebra_loader('train', config.batch_size)
    val_horse_loader, val_zebra_loader = get_horse2zebra_loader('test', config.batch_size)
    total_batch = min(len(train_horse_loader), len(train_zebra_loader))

    # Image Pool #
    masked_fake_A_pool = ImageMaskPool(config.pool_size)
    masked_fake_B_pool = ImageMaskPool(config.pool_size)

    # Prepare Networks #
    Attn_A = Attention()
    Attn_B = Attention()
    G_A2B = Generator()
    G_B2A = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    networks = [Attn_A, Attn_B, G_A2B, G_B2A, D_A, D_B]
    for network in networks:
        network.to(device)

    # Loss Function #
    criterion_Adversarial = nn.MSELoss()
    criterion_Cycle = nn.L1Loss()

    # Optimizers #
    D_optim = torch.optim.Adam(chain(D_A.parameters(), D_B.parameters()), lr=config.lr, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(chain(Attn_A.parameters(), Attn_B.parameters(), G_A2B.parameters(), G_B2A.parameters()), lr=config.lr, betas=(0.5, 0.999))

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_A_losses, D_B_losses = [], []
    G_A_losses, G_B_losses = [], []

    # Train #
    print("Training Unsupervised Attention-Guided GAN started with total epoch of {}.".format(config.num_epochs))

    for epoch in range(config.num_epochs):

        for i, (real_A, real_B) in enumerate(zip(train_horse_loader, train_zebra_loader)):

            # Data Preparation #
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            ###################
            # Train Generator #
            ###################

            set_requires_grad([D_A, D_B], requires_grad=False)

            # Adversarial Loss using real A #
            attn_A = Attn_A(real_A)
            fake_B = G_A2B(real_A)

            masked_fake_B = fake_B * attn_A + real_A * (1-attn_A)

            masked_fake_B *= attn_A
            prob_real_A = D_A(masked_fake_B)
            real_labels = torch.ones(prob_real_A.size()).to(device)

            G_loss_A = criterion_Adversarial(prob_real_A, real_labels)

            # Adversarial Loss using real B #
            attn_B = Attn_B(real_B)
            fake_A = G_B2A(real_B)

            masked_fake_A = fake_A * attn_B + real_B * (1-attn_B)

            masked_fake_A *= attn_B
            prob_real_B = D_B(masked_fake_A)
            real_labels = torch.ones(prob_real_B.size()).to(device)

            G_loss_B = criterion_Adversarial(prob_real_B, real_labels)

            # Cycle Consistency Loss using real A #
            attn_ABA = Attn_B(masked_fake_B)
            fake_ABA = G_B2A(masked_fake_B)
            masked_fake_ABA = fake_ABA * attn_ABA + masked_fake_B * (1 - attn_ABA)

            # Cycle Consistency Loss using real B #
            attn_BAB = Attn_A(masked_fake_A)
            fake_BAB = G_A2B(masked_fake_A)
            masked_fake_BAB = fake_BAB * attn_BAB + masked_fake_A * (1 - attn_BAB)

            # Cycle Consistency Loss #
            G_cycle_loss_A = config.lambda_cycle * criterion_Cycle(masked_fake_ABA, real_A)
            G_cycle_loss_B = config.lambda_cycle * criterion_Cycle(masked_fake_BAB, real_B)

            # Total Generator Loss #
            G_loss = G_loss_A + G_loss_B + G_cycle_loss_A + G_cycle_loss_B

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            #######################
            # Train Discriminator #
            #######################

            set_requires_grad([D_A, D_B], requires_grad=True)

            # Train Discriminator A using real A #
            prob_real_A = D_A(real_B)
            real_labels = torch.ones(prob_real_A.size()).to(device)
            D_loss_real_A = criterion_Adversarial(prob_real_A, real_labels)

            # Add Pooling #
            masked_fake_B, attn_A = masked_fake_B_pool.query(masked_fake_B, attn_A)
            masked_fake_B *= attn_A

            # Train Discriminator A using fake B #
            prob_fake_B = D_A(masked_fake_B.detach())
            fake_labels = torch.zeros(prob_fake_B.size()).to(device)
            D_loss_fake_A = criterion_Adversarial(prob_fake_B, fake_labels)

            D_loss_A = (D_loss_real_A + D_loss_fake_A).mean()

            # Train Discriminator B using real B #
            prob_real_B = D_B(real_A)
            real_labels = torch.ones(prob_real_B.size()).to(device)
            D_loss_real_B = criterion_Adversarial(prob_real_B, real_labels)

            # Add Pooling #
            masked_fake_A, attn_B = masked_fake_A_pool.query(masked_fake_A, attn_B)
            masked_fake_A *= attn_B

            # Train Discriminator B using fake A #
            prob_fake_A = D_B(masked_fake_A.detach())
            fake_labels = torch.zeros(prob_fake_A.size()).to(device)
            D_loss_fake_B = criterion_Adversarial(prob_fake_A, fake_labels)

            D_loss_B = (D_loss_real_B + D_loss_fake_B).mean()

            # Calculate Total Discriminator Loss #
            D_loss = D_loss_A + D_loss_B

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            # Add items to Lists #
            D_A_losses.append(D_loss_A.item())
            D_B_losses.append(D_loss_B.item())
            G_A_losses.append(G_loss_A.item())
            G_B_losses.append(G_loss_B.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("UAG-GAN | Epoch [{}/{}] | Iteration [{}/{}] | D A Losses {:.4f} | D B Losses {:.4f} | G A Losses {:.4f} | G B Losses {:.4f}".
                      format(epoch+1, config.num_epochs, i+1, total_batch, np.average(D_A_losses), np.average(D_B_losses), np.average(G_A_losses), np.average(G_B_losses)))

                # Save Sample Images #
                save_samples(val_horse_loader, val_zebra_loader, G_A2B, G_B2A, Attn_A, Attn_B, epoch, config.samples_path)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch + 1) % config.save_every == 0:
            torch.save(G_A2B.state_dict(), os.path.join(config.weights_path, 'UAG-GAN_Generator_A2B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(G_B2A.state_dict(), os.path.join(config.weights_path, 'UAG-GAN_Generator_B2A_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(Attn_A.state_dict(), os.path.join(config.weights_path, 'UAG-GAN_Attention_A_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(Attn_B.state_dict(), os.path.join(config.weights_path, 'UAG-GAN_Attention_B_Epoch_{}.pkl'.format(epoch+1)))

    # Make a GIF file #
    make_gifs_train("UAG-GAN", config.samples_path)

    # Plot Losses #
    plot_losses(D_A_losses, D_B_losses, G_A_losses, G_B_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()



