import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.ndf = 64
        self.out_channels = 1

        # Discriminator having last patch of (1, 14, 14)
        # (batch_size, 3, 128, 128) -> (batch_size, 1, 14, 14)
        self.main_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),

            nn.Conv2d(self.in_channels, self.ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 1, 1),
            nn.InstanceNorm2d(self.ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*4, self.out_channels, 4, 1, 1)
        )

        # Discriminator having last patch of (1, 30, 30)
        # (batch_size, 3, 128, 128) -> (batch_size, 1, 30, 30)
        self.main_2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 1, 1),
            nn.InstanceNorm2d(self.ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.out_channels, 4, 1, 1)
        )

    def forward(self, x):
        out_1 = self.main_1(x)
        out_2 = self.main_2(x)
        return out_1, out_2


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.short_cut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.main(x) + self.short_cut(x)
        return out


class Encoder(nn.Module):
    def __init__(self, z_dim=8):
        super(Encoder, self).__init__()

        self.in_channels = 3
        self.nef = 64

        self.main = nn.Sequential(                               # (batch_size, 3, 128, 128)
            nn.Conv2d(self.in_channels, self.nef, 4, 2, 1),      # (batch_size, 64, 64, 64)

            ResBlock(self.nef, self.nef*2),                      # (batch_size, 128, 32, 32)
            ResBlock(self.nef*2, self.nef*3),                    # (batch_size, 192, 16, 16)
            ResBlock(self.nef*3, self.nef*4),                    # (batch_size, 256, 8, 8)

            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=8, stride=8, padding=0)     # (batch_size, 256)
        )

        self.mu = nn.Linear(self.nef*4, z_dim)                   # (256, 8)
        self.log_var = nn.Linear(self.nef*4, z_dim)              # (256, 8)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)

        mu = self.mu(x)
        log_variance = self.log_var(x)

        return mu, log_variance


class Generator(nn.Module):
    def __init__(self, z_dim=8):
        super(Generator, self).__init__()

        self.in_channels = 3
        self.ngf = 64
        self.out_channels = 3

        self.down_layer_1 = nn.Sequential(
            nn.Conv2d(self.in_channels + z_dim, self.ngf, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_layer_2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 2, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_layer_3 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 4, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_layer_4 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_layer_5 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_layer_6 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_layer_7 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.up_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.ReLU()
        )

        self.up_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.ReLU()
        )

        self.up_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.ReLU()
        )

        self.up_layer_4 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 4, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 4, affine=True),
            nn.ReLU()
        )

        self.up_layer_5 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 2, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf * 2, affine=True),
            nn.ReLU()
        )

        self.up_layer_6 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf, 4, 2, 1),
            nn.InstanceNorm2d(self.ngf, affine=True),
            nn.ReLU()
        )

        self.up_layer_7 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2, self.out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, z):                                        # (batch_size, z_dim)
        z = z.unsqueeze(dim=2)                                      # (batch_size, z_dim, 1)
        z = z.unsqueeze(dim=3)                                      # (batch_size, z_dim, 1, 1)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))    # (batch_size, z_dim, H, W)
        x_z = torch.cat([x, z], dim=1)                              # (batch_size, 3+z_dim, H, W)

        down_1 = self.down_layer_1(x_z)
        down_2 = self.down_layer_2(down_1)
        down_3 = self.down_layer_3(down_2)
        down_4 = self.down_layer_4(down_3)
        down_5 = self.down_layer_5(down_4)
        down_6 = self.down_layer_6(down_5)
        down_7 = self.down_layer_7(down_6)

        up_1 = self.up_layer_1(down_7)
        up_2 = self.up_layer_2(torch.cat([up_1, down_6], dim=1))
        up_3 = self.up_layer_3(torch.cat([up_2, down_5], dim=1))
        up_4 = self.up_layer_4(torch.cat([up_3, down_4], dim=1))
        up_5 = self.up_layer_5(torch.cat([up_4, down_3], dim=1))
        up_6 = self.up_layer_6(torch.cat([up_5, down_2], dim=1))
        up_7 = self.up_layer_7(torch.cat([up_6, down_1], dim=1))

        return up_7
