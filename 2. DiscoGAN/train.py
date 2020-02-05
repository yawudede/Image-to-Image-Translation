import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from itertools import chain

from models import *
from edges2shoes import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(batch_size, num_epochs):

    # Path
    results_path = './data/results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Prepare Data Loader
    train_loader = get_edges2shoes_loader(batch_size=batch_size, purpose='train')
    total_batch = len(train_loader)
    val_loader = get_edges2shoes_loader(batch_size=8, purpose='val')

    # Models
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    networks = [D_A, D_B, G_A2B, G_B2A]

    for network in networks:
        network.apply(weights_init)

    # Criterion
    criterion_GAN = nn.BCELoss()
    criterion_Recon = nn.MSELoss()
    criterion_Feature = nn.HingeEmbeddingLoss()

    losses_D, losses_G = [], []

    # Optimizers
    optim_D = torch.optim.Adam(chain(D_A.parameters(), D_B.parameters()), lr=2e-4, betas=(0.5, 0.999),
                               weight_decay=0.00001)

    optim_G = torch.optim.Adam(chain(G_A2B.parameters(), G_B2A.parameters()), lr=2e-4, betas=(0.5, 0.999),
                               weight_decay=0.00001)

    # Training
    print("Training started with total batch of {}".format(total_batch))
    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):

            real_A = sample['A']
            real_B = sample['B']

            real_A = Variable(real_A).to(device)
            real_B = Variable(real_B).to(device)

            ### 0. Initialize
            D_A.zero_grad()
            D_B.zero_grad()
            G_A2B.zero_grad()
            G_B2A.zero_grad()

            ### 1. Foward Pass
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            real_out_A, real_features_A = D_A(real_A)
            fake_out_A, fake_features_A = D_A(fake_B)

            real_out_B, real_features_B = D_B(real_B)
            fake_out_B, fake_features_B = D_B(fake_A)

            fake_ABA = G_B2A(fake_B)
            fake_BAB = G_A2B(fake_A)

            real_labels = torch.FloatTensor(real_out_A.size()).fill_(1.0)
            real_labels = Variable(real_labels, requires_grad=False).to(device)

            fake_labels = torch.FloatTensor(fake_out_A.size()).fill_(0.0)
            fake_labels = Variable(fake_labels, requires_grad=False).to(device)

            ### 2. Train Discriminator
            # Discriminator A Loss
            loss_real_D_A = criterion_GAN(real_out_A, real_labels)
            loss_fake_D_A = criterion_GAN(fake_out_A, fake_labels)

            loss_D_A = (loss_real_D_A + loss_fake_D_A).mean()

            # Discriminator B Loss
            loss_real_D_B = criterion_GAN(real_out_B, real_labels)
            loss_fake_D_B = criterion_GAN(fake_out_B, fake_labels)

            loss_D_B = (loss_real_D_B + loss_fake_D_B).mean()

            # Total Loss
            loss_D = loss_D_A + loss_D_B

            # Back Propagation and Update
            loss_D.backward(retain_graph=True)
            optim_D.step()

            ### 3. Train Generator
            # Reconstruction Loss
            loss_recon_G_A = criterion_Recon(fake_ABA, real_A)
            loss_recon_G_B = criterion_Recon(fake_BAB, real_B)

            # GAN Loss
            loss_gan_G_A = criterion_GAN(fake_out_A, real_labels)
            loss_gan_G_B = criterion_GAN(fake_out_B, real_labels)

            # Feature Loss
            loss_feature_G_A = loss_feature(real_features_A, fake_features_A, criterion_Feature)
            loss_feature_G_B = loss_feature(real_features_B, fake_features_B, criterion_Feature)

            # Generator A and Generator B Loss
            loss_G_A = loss_recon_G_A*0.01 + (loss_gan_G_A*0.1 + loss_feature_G_A*0.9)*0.99
            loss_G_B = loss_recon_G_B*0.01 + (loss_gan_G_B*0.1 + loss_feature_G_B*0.9)*0.99

            # Total Loss
            loss_G = loss_G_A + loss_G_B

            # Back Propagation and Update
            loss_G.backward(retain_graph=True)
            optim_G.step()

            ### 4. Print Statistics
            if (i + 1) % 100 == 0:
                print("DiscoGAN | Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, num_epochs, i+1, total_batch, loss_D.item(), loss_G.item()))

                losses_D.append(loss_D.item())
                losses_G.append(loss_G.item())

    ### 5. Save Images
        sample_images(val_loader, epoch, G_A2B, G_B2A)

    make_gifs_train(results_path, "DiscoGAN")
    plot_losses(losses_D, losses_G, num_epochs, results_path)

    ### 6. Save Models
    torch.save(D_A.state_dict(), './data/results/DiscoGAN_Discriminator_A.pkl')
    torch.save(D_B.state_dict(), './data/results/DiscoGAN_Discriminator_B.pkl')
    torch.save(G_A2B.state_dict(), './data/results/DiscoGAN_Generator_A2B.pkl')
    torch.save(G_B2A.state_dict(), './data/results/DiscoGAN_Generator_B2A.pkl')

    print("Training finished.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    batch_size = 32
    num_epochs = 40
    train(batch_size, num_epochs)