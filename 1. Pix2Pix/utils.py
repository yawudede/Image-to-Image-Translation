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


def plot_losses(discriminator_losses, generator_losses, num_epochs, path):
    plt.figure(figsize = (10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("Pix2Pix Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_losses, label='Discriminator', alpha=0.5)
    plt.plot(generator_losses, label='Generator', alpha=0.5)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(path, 'Pix2Pix_Losses_Epoch_{}.png'.format(num_epochs)))


def sample_images(data_loader, epoch, generator, path):
    batch = next(iter(data_loader))
    input = Variable(batch['A'].type(torch.FloatTensor).to(device))
    target = Variable((batch['B']).type(torch.FloatTensor).to(device))

    generator.eval()
    generated = generator(input)

    result = torch.cat((target, input, generated), 0)
    result = ((result.data + 1) / 2).clamp(0, 1)
    save_image(result, os.path.join(path, 'Pix2Pix_Facades_Epoch_%03d.png' % (epoch + 1)), nrow=3, normalize=True)


def make_gifs_train(title, path):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Facades_Epoch_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Train_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def make_gifs_test(title, path):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Results_%03d.png' % (title, i + 1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Test_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))