import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(self.ndf*8, self.out_channel, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        feature = [layer2, layer3, layer4]

        return layer5, feature


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channel = 3
        self.ngf = 64
        self.out_channel = 3

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.ngf, self.ngf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.ngf*2, self.ngf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ngf*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.ngf*4, self.ngf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf, self.out_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out