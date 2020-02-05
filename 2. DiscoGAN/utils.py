from matplotlib import pyplot as plt
import os
import imageio
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def plot_losses(discriminator_losses, generator_losses, num_epochs, results_path):
    plt.figure(figsize = (10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("DiscoGAN Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_losses, label='Discriminator', alpha = 0.5)
    plt.plot(generator_losses, label='Generator', alpha = 0.5)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(os.path.join(results_path, 'DiscoGAN_Losses_Epoch_{}.png'.format(num_epochs)))


def make_gifs_sample(path, title):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Shoes_Epoch_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Results.gif'.format(title), generated_images, fps=5)
    print("{} Gif file is generated.".format(title))


def make_gifs_test(path, title):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Shoes_Results_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Results.gif'.format(title), generated_images, fps=5)
    print("{} Gif file is generated.".format(title))


def loss_feature(real_features, fake_features, criterion):
    losses = 0
    for real_feature, fake_feature in zip(real_features, fake_features):
        l2_loss = (real_feature.mean(0) - fake_feature.mean(0)) ** 2
        labels = Variable(torch.ones(l2_loss.size())).to(device)
        loss = criterion(l2_loss, labels)
        losses += loss
    return losses


def sample_images(data_loader, epoch, generator_1, generator_2):

    images = next(iter(data_loader))

    real_A = images['A'].to(device)
    fake_B = generator_1(real_A)
    fake_ABA = generator_2(fake_B)
    real_B = images['B'].to(device)
    fake_A = generator_2(real_B)
    fake_BAB = generator_1(fake_A)

    result = torch.cat((real_A, fake_A, fake_BAB, real_B, fake_B, fake_ABA), 0)
    result = ((result.data + 1) / 2).clamp(0, 1)
    save_image(result, './data/results/DiscoGAN_Edges2Shoes_Epoch_%03d.png'%(epoch + 1),
               nrow=8, normalize=True)