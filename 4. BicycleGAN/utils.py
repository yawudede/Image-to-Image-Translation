import os
import imageio
from matplotlib import pyplot as plt

import torch
from torch.nn import init
from torchvision.utils import save_image

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
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)


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


def set_requires_grad(networks, requires_grad=False):
    """Prevent a Network from Updating"""
    for network in networks:
        for param in network.parameters():
            param.requires_grad = requires_grad


def denorm(x):
    """De-normalization"""
    out = (x+1)/2
    return out.clamp(0, 1)


def sample_images(data_loader, generator, noise, epoch, num_images, path):
    """Save sample images for every epoch"""
    sketch, ground_truth = next(iter(data_loader))
    N = sketch.size(0)
    sketch = sketch.type(torch.FloatTensor).to(device)
    results = torch.FloatTensor(N * (1 + config.num_images), 3, config.crop_size, config.crop_size)

    for i in range(N):
        results[i * (1 + num_images)] = sketch[i].data

        for j in range(num_images):
            image = sketch[i].unsqueeze(dim=0)
            noise_to_generator = noise[i, j, :].unsqueeze(dim=0)

            out = generator(image, noise_to_generator)
            results[i * (1 + num_images) + j + 1] = out.data

    save_image(denorm(results.data),
               os.path.join(path, 'BicycleGAN_Edges2Handbags_Epoch_%03d.png' % (epoch + 1)),
               nrow=(1 + num_images),
               )


def make_gifs_train(title, path):
    """Create a GIF file after train"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Handbags_Epoch_001.png' % (title))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Train_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def make_gifs_test(title, path):
    """Create a GIF file after inference"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Handbags_Results_%03d.png' % (title, i + 1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Test_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def plot_losses(discriminator_losses, encoder_generator_losses, generator_losses, num_epochs, path):
    """Plot Losses After Training"""
    plt.figure(figsize=(10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("BicycleGAN Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_losses, label='Discriminator', alpha=0.5)
    plt.plot(encoder_generator_losses, label='Encoder & Generator', alpha=0.5)
    plt.plot(generator_losses, label='Generator', alpha=0.5)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(path, 'BicycleGAN_Losses_Epoch_{}.png'.format(num_epochs)))