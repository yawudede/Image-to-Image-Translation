import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import imageio


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def sample_images(data_loader, epoch, generator, noise, num_images, path):

    sketch, ground_truth = next(iter(data_loader))
    N = sketch.size(0)
    sketch = Variable(sketch.type(torch.FloatTensor)).to(device)
    results = torch.FloatTensor(N*(1 + num_images), 3, 128, 128)

    for i in range(N):
        results[i * (1 + num_images)] = sketch[i].data

        for j in range(num_images):
            image = sketch[i].unsqueeze(dim=0)
            noise_to_generator = noise[i, j, :].unsqueeze(dim=0)

            out = generator(image, noise_to_generator)
            results[i * (1 + num_images) + j + 1] = out.data

    results = results/2 + 0.5
    save_image(results, os.path.join(path, 'BicycleGAN_Edges2Handbags_Epoch_%03d.png'
               % (epoch + 1)), nrow=(1 + num_images), normalize=True)


def make_gifs_train(title, path):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Handbags_Epoch_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Train_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def make_gifs_test(title, path):
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Edges2Handbags_Results_%03d.png' % (title, i + 1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Test_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def plot_losses(discriminator_losses, encoder_generator_losses, generator_losses, num_epochs, path):
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