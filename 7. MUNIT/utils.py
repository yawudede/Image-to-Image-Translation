import os
import imageio
from matplotlib import pyplot as plt

import torch
from torch.nn import init
from torchvision.utils import save_image, make_grid

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def init_weights_normal(m):
    """Normal Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)


def init_weights_xavier(m):
    """Xavier Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)


def init_weights_kaiming(m):
    """Kaiming He Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


def get_lr_scheduler(optimizer):
    """Learning Rate Scheduler"""
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)
    elif config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def criterion_Adversarial(outs, labels):
    """Adversarial Loss"""
    loss = 0
    for i, out in enumerate(outs):
        if labels == 'real':
            loss += torch.mean((out - 1)**2)
        elif labels == 'fake':
            loss += torch.mean((out - 0)**2)
    return loss


def set_requires_grad(networks, requires_grad=False):
    """Prevent a Network from Updating"""
    for network in networks:
        for param in network.parameters():
            param.requires_grad = requires_grad


def denorm(x):
    """De-normalization"""
    out = (x+1) / 2
    return out.clamp(0, 1)


def sample_images(data_loader_1, data_loader_2, generator_1, generator_2, fixed_style_A, fixed_style_B, epoch, path):
    """Save Sample Images for Every Epoch"""

    generator_2.eval()
    generator_1.eval()

    real_A = next(iter(data_loader_1))
    real_B = next(iter(data_loader_2))

    real_A = real_A.to(device)
    real_B = real_B.to(device)

    style_A = torch.randn(real_A.size(0), config.style_dim, 1, 1).to(device)
    style_B = torch.randn(real_B.size(0), config.style_dim, 1, 1).to(device)

    real_A_recon, real_B_recon = list(), list()
    fake_A_1, fake_A_2 = list(), list()
    fake_B_1, fake_B_2 = list(), list()

    for i in range(real_A.size(0)):
        content_A, fake_style_A = generator_2.encode(real_A[i].unsqueeze(dim=0))
        content_B, fake_style_B = generator_1.encode(real_B[i].unsqueeze(dim=0))

        fake_A_1.append(generator_2.decode(content_B, style_A[i].unsqueeze(dim=0)))
        fake_A_2.append(generator_2.decode(content_B, fixed_style_A[i].unsqueeze(dim=0)))

        fake_B_1.append(generator_1.decode(content_A, style_B[i].unsqueeze(dim=0)))
        fake_B_2.append(generator_1.decode(content_A, fixed_style_B[i].unsqueeze(dim=0)))

        real_A_recon.append(generator_2.decode(content_A, fake_style_A))
        real_B_recon.append(generator_1.decode(content_B, fake_style_B))

    fake_A_1s = torch.cat(fake_A_1, dim=0)
    fake_A_2s = torch.cat(fake_A_2, dim=0)

    fake_B_1s = torch.cat(fake_B_1, dim=0)
    fake_B_2s = torch.cat(fake_B_2, dim=0)

    real_A_recons = torch.cat(real_A_recon, dim=0)
    real_B_recons = torch.cat(real_B_recon, dim=0)

    images = [real_A, fake_B_1s, fake_B_2s, real_A_recons, real_B, fake_A_1s, fake_A_2s, real_B_recons]
    result = torch.cat(images, dim=0)

    save_image(result.data,
               os.path.join(path, 'MUNIT_Edges2Shoes_Epoch_%03d.png' % (epoch + 1)),
               nrow=config.display_size,
               normalize=True)


def plot_losses(discriminator_losses, generator_losses, num_epochs, path):
    """Plot Losses After Training"""
    plt.figure(figsize=(10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("MUNIT Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_losses, label='Discriminator', alpha=0.5)
    plt.plot(generator_losses, label='Generator', alpha=0.5)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(path, 'MUNIT_Losses_Epoch_{}.png'.format(num_epochs)))


def make_gifs_train(title, path):
    """Create a GIF file After Train"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Shoes_Epoch_%03d.png.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Train_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def make_gifs_test(title, sort, path):
    """Make a GIF file After Inference"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Shoes_%s_Results_%03d.png' % (title, sort, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_{}_Test_Results.gif'.format(sort, title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))
