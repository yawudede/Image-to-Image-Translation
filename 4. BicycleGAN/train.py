import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from edges2handbags import *
from models import *
from utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(batch_size, num_epochs):

    # Results Path #
    results_path = './data/results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Prepare Data Loader #
    train_loader = get_edges2handbags_loader('train', batch_size)
    val_loader = get_edges2handbags_loader('val', batch_size)
    total_batch = len(train_loader)

    # Relevant Constants #
    test_size = 20
    z_dim = 8
    n_test_image = 5
    lambda_KL = 0.01
    lambda_Image = 10
    lambda_Z = 0.5

    # Networks #
    D_cVAE = Discriminator().to(device)
    D_cLR = Discriminator().to(device)
    E = Encoder(z_dim).to(device)
    G = Generator(z_dim).to(device)

    networks = [D_cVAE, D_cLR, E, G]
    for network in networks:
        network.apply(weights_init)

    # Criterion #
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    D_losses, E_G_losses, G_losses = [], [], []

    # Optimizers #
    optim_D_cVAE = torch.optim.Adam(D_cVAE.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D_cLR = torch.optim.Adam(D_cLR.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_E = torch.optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Fixed Noise #
    fixed_noise = Variable(torch.randn(test_size, n_test_image, z_dim).type(torch.FloatTensor)).to(device)

    # Training #
    print("Training BicycleGAN started with batch size of {} and total batch of {}.".format(batch_size, total_batch))
    for epoch in range(num_epochs):
        for i, (sketch, ground_truth) in enumerate(train_loader):

            # Data Preparation #
            sketch = Variable(sketch.type(torch.FloatTensor), requires_grad=True).to(device)
            ground_truth = Variable(ground_truth.type(torch.FloatTensor), requires_grad=True).to(device)

            # Separate Data for D_cVAE-GAN and D_cLR-GAN #
            cVAE_data = {'sketch': sketch[0].unsqueeze(dim=0), 'ground_truth': ground_truth[0].unsqueeze(dim=0)}
            cLR_data = {'sketch': sketch[1].unsqueeze(dim=0), 'ground_truth': ground_truth[1].unsqueeze(dim=0)}

            ### Train Discriminator ###
            ## Discriminator cVAE-GAN ##
            # Initialize #
            optim_D_cVAE.zero_grad()

            # Encode Latent Vector #
            mu, log_var = E(cVAE_data['ground_truth'])
            std = torch.exp(log_var / 2)
            random_z = Variable(torch.randn(1, z_dim).type(torch.FloatTensor), requires_grad=True).to(device)
            encoded_z = mu + (random_z * std)

            # Generate Fake Image #
            fake_image_cVAE = G(cVAE_data['sketch'], encoded_z)

            # Forward to Discriminator cVAE-GAN #
            prob_real_D_cVAE_1, prob_real_D_cVAE_2 = D_cVAE(cVAE_data['ground_truth'])
            prob_fake_D_cVAE_1, prob_fake_D_cVAE_2 = D_cVAE(fake_image_cVAE)

            # Real and Fake Labels #
            real_labels_1 = Variable(torch.ones(prob_real_D_cVAE_1.size()).type(torch.FloatTensor),
                                     requires_grad=False).to(device)
            real_labels_2 = Variable(torch.ones(prob_real_D_cVAE_2.size()).type(torch.FloatTensor),
                                     requires_grad=False).to(device)

            fake_labels_1 = Variable(torch.zeros(prob_fake_D_cVAE_1.size()).type(torch.FloatTensor),
                                     requires_grad=False).to(device)
            fake_labels_2 = Variable(torch.zeros(prob_fake_D_cVAE_2.size()).type(torch.FloatTensor),
                                     requires_grad=False).to(device)


            # Adversarial Loss #
            D_cVAE_1_real_loss = MSE_loss(prob_real_D_cVAE_1, real_labels_1)
            D_cVAE_1_fake_loss = MSE_loss(prob_fake_D_cVAE_1, fake_labels_1)

            D_cVAE_1_loss = D_cVAE_1_real_loss + D_cVAE_1_fake_loss

            D_cVAE_2_real_loss = MSE_loss(prob_real_D_cVAE_2, real_labels_2)
            D_cVAE_2_fake_loss = MSE_loss(prob_fake_D_cVAE_2, fake_labels_2)

            D_cVAE_2_loss = D_cVAE_2_real_loss + D_cVAE_2_fake_loss

            ## Discriminator cLR ##
            # Initialize #
            optim_D_cLR.zero_grad()

            # Random Latent Vector #
            random_z = Variable(torch.randn(1, z_dim).type(torch.FloatTensor), requires_grad=True).to(device)

            # Generate Fake Image #
            fake_image_cLR = G(cLR_data['sketch'], random_z)

            # Forward to Discriminator cLR-GAN #
            prob_real_D_cLR_1, prob_real_D_cLR_2 = D_cLR(cLR_data['ground_truth'])
            prob_fake_D_cLR_1, prob_fake_D_cLR_2 = D_cLR(fake_image_cLR)

            # Adversarial Loss #
            D_cLR_1_real_loss = MSE_loss(prob_real_D_cLR_1, real_labels_1)
            D_cLR_1_fake_loss = MSE_loss(prob_fake_D_cLR_1, fake_labels_1)

            D_cLR_1_loss = D_cLR_1_real_loss + D_cLR_1_fake_loss

            D_cLR_2_real_loss = MSE_loss(prob_real_D_cLR_2, real_labels_2)
            D_cLR_2_fake_loss = MSE_loss(prob_fake_D_cLR_2, fake_labels_2)

            D_cLR_2_loss = D_cLR_2_real_loss + D_cLR_2_fake_loss

            # Total Discriminator Loss #
            D_loss = D_cVAE_1_loss + D_cVAE_2_loss + D_cLR_1_loss + D_cLR_2_loss

            # Back Propagation and Update #
            D_loss.backward(retain_graph=True)
            optim_D_cVAE.step()
            optim_D_cLR.step()

            ### Train Generator and Encoder ###
            ## GAN Loss ##
            # Initialize #
            optim_G.zero_grad()
            optim_E.zero_grad()

            # Encode Latent Vector #
            mu, log_var = E(cVAE_data['ground_truth'])
            std = torch.exp(log_var/2)
            random_z = Variable(torch.randn(1, z_dim).type(torch.FloatTensor), requires_grad=False).to(device)
            encoded_z = mu + (std * random_z)

            # Generate Fake Image #
            fake_image_cVAE = G(cVAE_data['sketch'], encoded_z)

            # Adversarial Loss of cVAE #
            prob_fake_D_cVAE_1, prob_fake_D_cVAE_2 = D_cVAE(fake_image_cVAE)

            G_GAN_cVAE_1_loss = MSE_loss(prob_fake_D_cVAE_1, real_labels_1)
            G_GAN_cVAE_2_loss = MSE_loss(prob_fake_D_cVAE_2, real_labels_2)

            G_GAN_cVAE_loss = G_GAN_cVAE_1_loss + G_GAN_cVAE_2_loss

            # Random Latent Vector #
            random_z = Variable(torch.randn(1, z_dim).type(torch.FloatTensor), requires_grad=False).to(device)

            # Generate Fake Image #
            fake_image_cLR = G(cLR_data['sketch'], random_z)

            # Adversarial Loss of cLR #
            prob_fake_D_cLR_1, prob_fake_D_cLR_2 = D_cLR(fake_image_cLR)

            G_GAN_cLR_1_loss = MSE_loss(prob_fake_D_cLR_1, real_labels_1)
            G_GAN_cLR_2_loss = MSE_loss(prob_fake_D_cLR_2, real_labels_2)

            G_GAN_cLR_loss = G_GAN_cLR_1_loss + G_GAN_cLR_2_loss

            G_GAN_loss = G_GAN_cVAE_loss + G_GAN_cLR_loss

            ## KL Divergence with N~(0, 1) ##
            KL_div = lambda_KL * torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

            ## Reconstruction of ground truth ##
            reconstruction_loss = lambda_Image * L1_loss(fake_image_cVAE, cVAE_data['ground_truth'])

            # Total Encoder and Generator Loss ##
            E_G_loss = G_GAN_loss + KL_div + reconstruction_loss

            # Back Propagation and Update #
            E_G_loss.backward(retain_graph=True)
            optim_E.step()
            optim_G.step()

            ### Train Generator only ###
            # Initialize #
            optim_G.zero_grad()
            mu, log_var = E(fake_image_cLR)
            recon_loss = L1_loss(mu, random_z)

            # Total Loss #
            G_loss = lambda_Z * recon_loss

            # Back Propagation and Update #
            G_loss.backward(retain_graph=True)
            optim_G.step()

            ### Print Statistics ###
            if (i+1) % 100 == 0:
                print("BicycleGAN | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | E&G Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, num_epochs, i+1, total_batch, D_loss.item(), E_G_loss.item(), G_loss.item()))

                D_losses.append(D_loss.item())
                E_G_losses.append(E_G_loss.item())
                G_losses.append(G_loss.item())

    # Save Images #
        sample_images(val_loader, epoch, G, fixed_noise, n_test_image, results_path)

    make_gifs_train("BicycleGAN", results_path)
    plot_losses(D_losses, E_G_losses, G_losses, num_epochs, results_path)

    # Save Models #
    torch.save(D_cVAE.state_dict(), './data/results/BicycleGAN_Discriminator_cVAE.pkl')
    torch.save(D_cLR.state_dict(), './data/results/BicycleGAN_Discriminator_cLR.pkl')
    torch.save(E.state_dict(), './data/results/BicycleGAN_Encoder.pkl')
    torch.save(G.state_dict(), './data/results/BicycleGAN_Generator.pkl')

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()

    batch_size = 2
    num_epochs = 1 # Increase it to at least 10 if you have enough GPU resources
    train(batch_size, num_epochs)


