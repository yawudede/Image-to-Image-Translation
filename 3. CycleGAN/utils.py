from matplotlib import pyplot as plt
import imageio
import os
import torch
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def plot_losses(discriminator_a_losses, discriminator_b_losses, generator_losses, num_epochs, results_path):
    plt.figure(figsize=(10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("CycleGAN Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_a_losses, label='Discriminator A', alpha=0.5)
    plt.plot(discriminator_b_losses, label='Discriminator B', alpha=0.5)
    plt.plot(generator_losses, label='Generator', alpha=0.5)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(results_path, 'CycleGAN_Losses_Epoch_{}.png'.format(num_epochs)))


def make_gifs_sample(path, title):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Horse2Zebra_Epoch_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Results_Sample.gif'.format(title), generated_images, fps=5)
    print("{} Gif file is generated.".format(title))


def make_gifs_test(path, title):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Horse2Zebra_Results_%03d.png' % (title, i + 1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Results_Test.gif'.format(title), generated_images, fps=5)
    print("{} Gif file is generated.".format(title))


def sample_images(horse_loader, zebra_loader, epoch, generator_1, generator_2):

    horse = next(iter(horse_loader))
    zebra = next(iter(zebra_loader))

    generator_1.eval()
    generator_2.eval()

    real_A = horse.to(device)
    fake_B = generator_1(real_A)
    fake_ABA = generator_2(fake_B)

    real_B = zebra.to(device)
    fake_A = generator_2(real_B)
    fake_BAB = generator_1(fake_A)

    result = torch.cat((real_A, fake_B, fake_ABA, real_B, fake_A, fake_BAB), 0)
    result = ((result.data + 1) / 2).clamp(0, 1)
    save_image(result, './data/results/CycleGAN_Horse2Zebra_Epoch_%03d.png'%(epoch + 1), nrow=6, normalize=True)
