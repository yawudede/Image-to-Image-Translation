import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.ndf = 64
        self.out_channels = 1

        # Discriminator having last patch of (1, 13, 13)
        # (batch_size, 3, 128, 128) -> (batch_size, 1, 13, 13)

        self.main_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),

            nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, self.out_channels, kernel_size=4, stride=1, padding=1, bias=False)
        )

        # Discriminator having last patch of (1, 30, 30)
        # (batch_size, 3, 128, 128) -> (batch_size, 1, 30, 30)

        self.main_2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf * 2, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf * 4, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.ndf * 4, self.out_channels, kernel_size=4, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out_1 = self.main_1(x)
        out_2 = self.main_2(x)
        return out_1, out_2


class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.InstanceNorm2d(in_features, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.short_cut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        out = self.main(x) + self.short_cut(x)
        return out


class Encoder(nn.Module):
    """Encoder Network"""
    def __init__(self, z_dim=8):
        super(Encoder, self).__init__()

        self.in_channels = 3
        self.nef = 64

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, self.nef, kernel_size=4, stride=2, padding=1),

            ResBlock(self.nef, self.nef*2),
            ResBlock(self.nef*2, self.nef*3),
            ResBlock(self.nef*3, self.nef*4),

            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        )

        self.fc_mean = nn.Linear(self.nef*4, z_dim)
        self.fc_std = nn.Linear(self.nef*4, z_dim)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)

        mean = self.fc_mean(x)
        log_var = self.fc_std(x)
        std = torch.exp(log_var / 2)

        return mean, std


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self, z_dim=8):
        super(Generator, self).__init__()

        self.in_channels = 3
        self.ngf = 64
        self.out_channels = 3

        self.down_1 = nn.Sequential(
            nn.Conv2d(self.in_channels + z_dim, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 2, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_3 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 4, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_4 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_5 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_6 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.down_7 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.ReLU()
        )

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.ReLU()
        )

        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 8, affine=True),
            nn.ReLU()
        )

        self.up_4 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 4,kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 4, affine=True),
            nn.ReLU()
        )

        self.up_5 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf * 2, affine=True),
            nn.ReLU()
        )

        self.up_6 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ngf, affine=True),
            nn.ReLU()
        )

        self.up_7 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, z):                                        # (batch_size, z_dim)
        z = z.unsqueeze(dim=2)                                      # (batch_size, z_dim, 1)
        z = z.unsqueeze(dim=3)                                      # (batch_size, z_dim, 1, 1)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))    # (batch_size, z_dim, H, W)
        x_z = torch.cat([x, z], dim=1)                              # (batch_size, 3+z_dim, H, W)

        down1 = self.down_1(x_z)
        down2 = self.down_2(down1)
        down3 = self.down_3(down2)
        down4 = self.down_4(down3)
        down5 = self.down_5(down4)
        down6 = self.down_7(down5)

        up1 = self.up_1(down6)
        up3 = self.up_3(torch.cat([up1, down5], dim=1))
        up4 = self.up_4(torch.cat([up3, down4], dim=1))
        up5 = self.up_5(torch.cat([up4, down3], dim=1))
        up6 = self.up_6(torch.cat([up5, down2], dim=1))
        out = self.up_7(torch.cat([up6, down1], dim=1))

        return out