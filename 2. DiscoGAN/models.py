import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channel = 3
        self.ngf = 64
        self.out_channel = 3

        self.main = nn.Sequential(
            # [-1, 3, 64x64] -> [-1, 64, 32x32]
            nn.Conv2d(self.in_channel, self.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 128, 16x16]
            nn.Conv2d(self.ngf, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 256, 8x8]
            nn.Conv2d(self.ngf*2, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 512, 4x4]
            nn.Conv2d(self.ngf*4, self.ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 256, 8x8]
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            # [-1, 128, 16x16]
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            # [-1, 256, 32x32]
            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # [-1, 3, 64x64]
            nn.ConvTranspose2d(self.ngf, self.out_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        self.layer1 = nn.Sequential(
            # [-1, 3, 64x64] -> [-1, 64, 32x32]
            nn.Conv2d(self.in_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            # [-1, 128, 16x16]
            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            # [-1, 256, 8x8]
            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            # [-1, 512, 4x4]
            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            # [-1, 1, 1x1]
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