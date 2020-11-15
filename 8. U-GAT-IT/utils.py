import os
import imageio
from matplotlib import pyplot as plt
import numpy as np
import cv2

import torch
from torch.nn import init

from config import *

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RhoClipper(object):
    """Clipping Rho"""
    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max
        assert clip_min < clip_max

    def __call__(self, module):
        if hasattr(module, "rho"):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


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


def set_requires_grad(networks, requires_grad=False):
    """Prevent a Network from Updating"""
    for network in networks:
        for param in network.parameters():
            param.requires_grad = requires_grad


def denorm(x):
    """De-normalization"""
    out = (x+1) / 2
    return out


def cam(x, image_size):
    """Visualize Grad-CAM"""
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (image_size, image_size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    cam_img = cam_img / 255.0
    return cam_img


def tensor2npy(x):
    """Convert Tensor to Numpy"""
    out = x.detach().cpu().numpy().transpose(1, 2, 0)
    return out


def RGB2BGR(x):
    """Convert RGB to BGR"""
    out = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return out


def save_samples(data_loader, generator_1, epoch, path):
    """Save Samples for Every Epoch during Training"""
    A2B = np.zeros((config.crop_size * 3, 0, 3))

    with torch.no_grad():
        for i in range(config.val_batch_size):

            # Prepare Data #
            real_A = next(iter(data_loader))
            real_A = real_A.to(device)

            # Generate Fake Images #
            fake_B, _, fake_B_heatmap = generator_1(real_A)

            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2npy(denorm(real_A[0]))),
                                                       cam(tensor2npy(fake_B_heatmap[0]), config.crop_size),
                                                       RGB2BGR(tensor2npy(denorm(fake_B[0])))), 0)), 1)

    cv2.imwrite(os.path.join(path, 'U-GAT-IT_Samples_Epoch_%03d.png' % (epoch + 1)), A2B * 255.0)


def plot_losses(discriminator_losses, generator_losses, num_epochs, path):
    """Plot Losses After Training"""
    plt.figure(figsize=(10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("U-GAT-IT Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_losses, label='Discriminator', alpha=0.5)
    plt.plot(generator_losses, label='Generator', alpha=0.5)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(path, 'U-GAT-IT_Losses_Epoch_{}.png'.format(num_epochs)))


def make_gifs_train(title, path):
    """Create a GIF file After Train"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Samples_Epoch_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Train_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def make_gifs_test(title, sort, path):
    """Make a GIF file After Inference"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_%s_Results_%03d.png' % (title, sort, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_{}_Test_Results.gif'.format(sort, title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))