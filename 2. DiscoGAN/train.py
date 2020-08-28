import os
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from config import *
from edges2shoes import get_edges2shoes_loader
from models import Discriminator, Generator
from utils import make_dirs, get_lr_scheduler, feature_loss, sample_images, plot_losses, make_gifs_train


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
    train_loader = get_edges2shoes_loader('train', config.batch_size)
    val_loader = get_edges2shoes_loader('val', config.val_batch_size)
    total_batch = len(train_loader)

    # Prepare Networks #
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Loss Function #
    criterion_Adversarial = nn.BCELoss()
    criterion_Recon = nn.MSELoss()
    criterion_Feature = nn.HingeEmbeddingLoss()

    # Optimizers #
    G_optim = torch.optim.Adam(chain(G_A2B.parameters(), G_B2A.parameters()), config.lr, betas=(0.5, 0.999), weight_decay=0.00001)
    D_optim = torch.optim.Adam(chain(D_A.parameters(), D_B.parameters()), config.lr, betas=(0.5, 0.999), weight_decay=0.00001)

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Lists #
    D_losses, G_losses = [], []

    # Constants #
    iters = 0

    # Training #
    print("Training DiscoGAN started with total epoch of {}.".format(config.num_epochs))

    for epoch in range(config.num_epochs):

        for i, (real_A, real_B) in enumerate(train_loader):

            # Data Preparation #
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Initialize Models #
            G_A2B.zero_grad()
            G_B2A.zero_grad()
            D_A.zero_grad()
            D_B.zero_grad()

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            ################
            # Forward Data #
            ################

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            prob_real_A, A_real_features = D_A(real_A)
            prob_fake_A, A_fake_features = D_A(fake_A)

            prob_real_B, B_real_features = D_B(real_B)
            prob_fake_B, B_fake_features = D_B(fake_B)

            #######################
            # Train Discriminator #
            #######################

            # Discriminator A #
            real_labels = Variable(torch.ones(prob_real_A.size()), requires_grad=False).to(device)
            D_real_loss_A = criterion_Adversarial(prob_real_A, real_labels)

            fake_labels = Variable(torch.zeros(prob_fake_A.size()), requires_grad=False).to(device)
            D_fake_loss_A = criterion_Adversarial(prob_fake_A, fake_labels)

            D_loss_A = (D_real_loss_A + D_fake_loss_A).mean()

            # Discriminator B #
            real_labels = Variable(torch.ones(prob_real_B.size()), requires_grad=False).to(device)
            D_real_loss_B = criterion_Adversarial(prob_real_B, real_labels)

            fake_labels = Variable(torch.zeros(prob_fake_B.size()), requires_grad=False).to(device)
            D_fake_loss_B = criterion_Adversarial(prob_fake_B, fake_labels)

            D_loss_B = (D_real_loss_B + D_fake_loss_B).mean()

            # Calculate Total Discriminator Loss #
            D_loss = D_loss_A + D_loss_B

            ###################
            # Train Generator #
            ###################

            # Adversarial Loss #
            real_labels = Variable(torch.ones(prob_real_A.size()), requires_grad=False).to(device)
            G_adv_loss_A = criterion_Adversarial(prob_fake_A, real_labels)

            real_labels = Variable(torch.ones(prob_real_B.size()), requires_grad=False).to(device)
            G_adv_loss_B = criterion_Adversarial(prob_fake_B, real_labels)

            # Feature Loss #
            G_feature_loss_A = feature_loss(criterion_Feature, A_real_features, A_fake_features)
            G_feature_loss_B = feature_loss(criterion_Feature, B_real_features, B_fake_features)

            # Reconstruction Loss #
            fake_ABA = G_B2A(fake_B)
            fake_BAB = G_A2B(fake_A)

            G_recon_loss_A = criterion_Recon(fake_ABA, real_A)
            G_recon_loss_B = criterion_Recon(fake_BAB, real_B)

            if iters < config.decay_gan_loss:
                rate = config.starting_rate
            else:
                print("Now the rate is changed to {}".format(config.changed_rate))
                rate = config.changed_rate

            G_loss_A = (G_adv_loss_A*0.1 + G_feature_loss_A*0.9) * (1.-rate) + G_recon_loss_A * rate
            G_loss_B = (G_adv_loss_B*0.1 + G_feature_loss_B*0.9) * (1.-rate) + G_recon_loss_B * rate

            # Calculate Total Generator Loss #
            G_loss = G_loss_A + G_loss_B

            # Back Propagation and Update #
            if iters % config.num_train_gen == 0:
                D_loss.backward()
                D_optim.step()
            else:
                G_loss.backward()
                G_optim.step()

            # Add items to Lists #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("DiscoGAN | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch + 1, config.num_epochs, i + 1, total_batch, np.average(D_losses), np.average(G_losses)))

                # Save Sample Images #
                sample_images(val_loader, G_A2B, G_B2A, epoch, config.samples_path)

            iters += 1

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch + 1) % config.save_every == 0:
            torch.save(G_A2B.state_dict(), os.path.join(config.weights_path, 'DiscoGAN_Generator_A2B_Epoch_{}.pkl'.format(epoch+1)))
            torch.save(G_B2A.state_dict(), os.path.join(config.weights_path, 'DiscoGAN_Generator_B2A_Epoch_{}.pkl'.format(epoch+1)))

    # Make a GIF file #
    make_gifs_train('DiscoGAN', config.samples_path)

    # Plot Losses #
    plot_losses(D_losses, G_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    train()