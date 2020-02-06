import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1


        model = [
            nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        ]

        n_blocks = 3

        for i in range(n_blocks):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ndf * mult * 2),
                nn.LeakyReLU(0.2)
            ]

        model += [nn.Conv2d(self.ndf * mult * 2, self.out_channel, kernel_size=4, stride=1, padding=1, bias=True),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channel = 3
        self.ngf = 64
        self.out_channel = 3
        self.num_residual_blocks = 9

        # initial block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.in_channel, self.ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(self.ngf),
            nn.ReLU()
        ]

        # down sampling
        n_downsampling = 3

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ngf * mult * 2),
                nn.ReLU()
            ]

        # residual blocks
        mult = 2 ** n_downsampling

        for i in range(self.num_residual_blocks):
            model += [ResidualBlock(self.ngf * mult)]

        # up sampling
        n_upsampling = 3

        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=True),
                nn.InstanceNorm2d(int(self.ngf * mult / 2)),
                nn.ReLU()
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.ngf, self.out_channel, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out