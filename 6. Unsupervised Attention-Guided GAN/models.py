import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False)
        )

    def forward(self, x):
        out = self.main(x)
        out += x
        return out


class Attention(nn.Module):
    """Attention Network"""
    def __init__(self):
        super(Attention, self).__init__()

        self.in_channels = 3
        self.naf = 64
        self.out_channels = 1

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, self.naf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(self.naf, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.naf, self.naf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.naf * 2),
            nn.ReLU(inplace=True),

            ResidualBlock(self.naf * 2),
            nn.ReLU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(self.naf * 2, self.naf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.naf * 2, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(self.naf * 2, self.naf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.naf, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.naf, self.out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        model = [
            nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        n_blocks = 2

        for i in range(n_blocks):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(self.ndf * mult * 2, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        model += [
            nn.Conv2d(self.ndf * mult * 2, self.out_channel, kernel_size=4, stride=1, padding=1, bias=False)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
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
            nn.Conv2d(self.in_channel, self.ngf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(self.ngf, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        ]

        # Down Sampling #
        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(self.ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True)
            ]

        # Residual Blocks #
        mult = 2 ** n_downsampling

        for i in range(self.num_residual_blocks):
            model += [ResidualBlock(self.ngf * mult)]

        # Up Sampling #
        n_upsampling = 2

        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(int(self.ngf * mult / 2), affine=False, track_running_stats=False),
                nn.ReLU(inplace=True)
            ]

        model += [
            nn.Conv2d(self.ngf, self.out_channel, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out