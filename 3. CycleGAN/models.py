import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        model = [
            nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        n_blocks = 3

        for i in range(n_blocks):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ndf * mult * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        model += [nn.Conv2d(self.ndf * mult * 2, self.out_channel, kernel_size=4, stride=1, padding=1, bias=True),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = self.relu(out)
        return out


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channel = 3
        self.ngf = 64
        self.out_channel = 3
        self.num_residual_blocks = 9

        # Initial Block #
        model = [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.in_channel, self.ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(self.ngf),
            nn.ReLU(inplace=True)
        ]

        # Down Sampling #
        n_downsampling = 3

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        # Residual Blocks #
        mult = 2 ** n_downsampling

        for i in range(self.num_residual_blocks):
            model += [ResidualBlock(self.ngf * mult)]

        # Up Sampling #
        n_upsampling = 3

        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=True),
                nn.InstanceNorm2d(int(self.ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]

        model += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.ngf, self.out_channel, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out