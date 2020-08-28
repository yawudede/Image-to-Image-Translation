import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator Network"""

    def __init__(self, num_classes=5):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.ndf = 64
        self.image_size = 128
        self.num_blocks = 6

        layers = []
        layers += [
            nn.Conv2d(self.in_channels, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True)
        ]

        curr_dim = self.ndf
        for i in range(1, self.num_blocks):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.01, inplace=True)
            ]
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)

        kernel_size = int(self.image_size / 2 ** self.num_blocks)
        self.conv_src = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(curr_dim, num_classes, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        out = self.main(x)
        out_src = self.conv_src(out)
        out_cls = self.conv_cls(out)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        out = x + self.main(x)
        return out


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self, num_classes=5):
        super(Generator, self).__init__()

        self.in_channels = 3
        self.n_sampling = 2
        self.ngf = 64
        self.num_blocks = 6
        self.out_channels = 3

        layers = []
        layers += [
            nn.Conv2d(self.in_channels + num_classes, self.ngf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(self.ngf, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ]

        # Down Sampling #
        curr_dim = self.ngf
        for i in range(self.n_sampling):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        # Residual Blocks #
        for j in range(self.num_blocks):
            layers += [
                ResidualBlock(curr_dim)
            ]

        # Up Sampling #
        for k in range(self.n_sampling):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        layers += [
            nn.Conv2d(curr_dim, self.out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), dim=1)
        out = self.main(x)
        return out