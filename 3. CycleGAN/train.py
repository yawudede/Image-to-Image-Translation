import torch
import torch.nn as nn
import torchvision
from itertools import chain
import os

from horse2zebra import *
from models import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(batch_size, num_epochs):

    # Results Path #
    results_path = './data/results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Prepare Data Loader #
    trainloader_horse, trainloader_zebra = get_horse2zebra_loader('train', batch_size)
    testloader_horse, testloader_zebra = get_horse2zebra_loader('test', 1)
    total_batch = min(len(trainloader_horse), len(trainloader_zebra))

    # Networks #
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    networks = [D_A, D_B, G_A2B, G_B2A]

    for network in networks:
        network.apply(weights_init)

    # Criterion #
    criterion_MSE = nn.MSELoss()
    criterion_Cycle = nn.L1Loss()
    criterion_Identity = nn.L1Loss()

    lambda_identity = 5
    lambda_cycle = 10

    losses_D_A, losses_D_B, losses_G = [], [], []

    # Optimizers #
    optim_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_G = torch.optim.Adam(chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))

    # Training #
    print("Training CycleGAN started with batch size of {} and total batch of {}.".format(batch_size, total_batch))
    for epoch in range(num_epochs):
        for i, (horse, zebra) in enumerate(zip(trainloader_horse, trainloader_zebra)):

            # Data Preparation #
            real_A, real_B = horse.to(device), zebra.to(device)

            ### Train Generator ###
            # Initialize #
            optim_G.zero_grad()

            # Identity Loss #
            identity_A = G_B2A(real_A)
            loss_identity_A = criterion_Identity(identity_A, real_A) * lambda_identity

            identity_B = G_A2B(real_B)
            loss_identity_B = criterion_Identity(identity_B, real_B) * lambda_identity

            # Mean Squared Error Loss (Transformation Loss) #
            fake_A = G_B2A(real_B)
            prob_fake_A = D_A(fake_A)
            real_labels = torch.ones(prob_fake_A.size()).to(device)
            loss_mse_B2A = criterion_MSE(prob_fake_A, real_labels)

            fake_B = G_A2B(real_A)
            prob_fake_B = D_B(fake_B)
            real_labels = torch.ones(prob_fake_B.size()).to(device)
            loss_mse_A2B = criterion_MSE(prob_fake_B, real_labels)

            # Cycle Loss #
            reconstructed_A = G_B2A(fake_B)
            loss_cycle_ABA = criterion_Cycle(reconstructed_A, real_A) * lambda_cycle

            reconstructed_B = G_A2B(fake_A)
            loss_cycle_BAB = criterion_Cycle(reconstructed_B, real_B) * lambda_cycle

            # Total Generator Loss #
            loss_G = loss_identity_A + loss_identity_B + loss_mse_A2B + loss_mse_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Back Propagation and Update #
            loss_G.backward()
            optim_G.step()

            ### Train Discriminator A ###
            # Initialize #
            optim_D_A.zero_grad()

            # Real Loss #
            prob_real_A = D_A(real_A)
            real_labels = torch.ones(prob_real_A.size()).to(device)
            loss_real_A = criterion_MSE(prob_real_A, real_labels)

            # Fake Loss
            fake_A = G_B2A(real_B)
            prob_fake_A = D_A(fake_A)
            fake_labels = torch.zeros(prob_fake_A.size()).to(device)
            loss_fake_A = criterion_MSE(prob_fake_A, fake_labels)

            # Total Loss
            loss_D_A = (loss_real_A + loss_fake_A).mean() * lambda_identity

            # Back propagation and Update
            loss_D_A.backward()
            optim_D_A.step()

            ### Train Discriminator B ###
            # Initialize #
            optim_D_B.zero_grad()

            # Real Loss #
            prob_real_B = D_B(real_B)
            real_labels = torch.ones(prob_real_B.size()).to(device)
            loss_real_B = criterion_MSE(prob_real_B, real_labels)

            # Fake Loss #
            fake_B = G_A2B(real_A)
            prob_fake_B = D_B(fake_B)
            fake_labels = torch.zeros(prob_fake_B.size()).to(device)
            loss_fake_B = criterion_MSE(prob_fake_B, fake_labels)

            # Total Discriminator Loss #
            loss_D_B = (loss_real_B + loss_fake_B).mean() * lambda_identity

            # Back propagation and Update #
            loss_D_B.backward()
            optim_D_B.step()

            ### Print Statistics ###
            if (i+1) % 100 == 0:
                print("CycleGAN | Epoch [{}/{}] | Iterations [{}/{}] | D_A Loss {:.4f} | D_B Loss {:.4f} | G Loss {:.4f}".
                      format(epoch + 1, num_epochs, i + 1, total_batch, loss_D_A.item(), loss_D_B.item(), loss_G.item()))

                losses_D_A.append(loss_D_A.item())
                losses_D_B.append(loss_D_B.item())
                losses_G.append(loss_G.item())

    # Save Images #
        sample_images(testloader_horse, testloader_zebra, epoch, G_A2B, G_B2A, results_path)

    make_gifs_train("CycleGAN", results_path)
    plot_losses(losses_D_A, losses_D_B, losses_G, num_epochs, results_path)

    # Save Models #
    torch.save(D_A.state_dict(), './data/results/CycleGAN_Discriminator_A.pkl')
    torch.save(D_B.state_dict(), './data/results/CycleGAN_Discriminator_B.pkl')
    torch.save(G_A2B.state_dict(), './data/results/CycleGAN_Generator_A2B.pkl')
    torch.save(G_B2A.state_dict(), './data/results/CycleGAN_Generator_B2A.pkl')

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()

    batch_size = 1
    num_epochs = 100
    train(batch_size, num_epochs)