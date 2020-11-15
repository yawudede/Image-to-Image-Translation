import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self, num_layers=5):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.dim = 64
        self.out_channels = 1

        model = []
        model += [
            nn.ReflectionPad2d(padding=1),
            spectral_norm(nn.Conv2d(self.in_channels, self.dim, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(1, num_layers - 2):
            factor = 2 ** (i-1)
            model += [
                nn.ReflectionPad2d(padding=1),
                spectral_norm(nn.Conv2d(self.dim * factor, self.dim * factor * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        factor = 2 ** (num_layers - 2 - 1)
        model += [
            nn.ReflectionPad2d(padding=1),
            spectral_norm(nn.Conv2d(self.dim * factor, self.dim * factor * 2, kernel_size=4, stride=1, padding=0, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        self.main = nn.Sequential(*model)

        # Class Activation Map (CAM) #
        factor = 2 ** (num_layers - 2)
        self.gap_fc = spectral_norm(nn.Linear(self.dim * factor, 1, bias=False))
        self.gmp_fc = spectral_norm(nn.Linear(self.dim * factor, 1, bias=False))

        self.heatmap = nn.Sequential(
            nn.Conv2d(self.dim * factor * 2, self.dim * factor, kernel_size=1, stride=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            spectral_norm(nn.Conv2d(self.dim * factor, self.out_channels, kernel_size=4, stride=1, padding=0, bias=False))
        )

    def forward(self, image):
        x = self.main(image)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logits = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(dim=2).unsqueeze(dim=3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logits = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(dim=2).unsqueeze(dim=3)

        cam_logits = torch.cat([gap_logits, gmp_logits], dim=1)
        x = torch.cat([gap, gmp], dim=1)
        x = self.heatmap(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)
        out = self.fc(x)

        return out, cam_logits, heatmap


class ILN(nn.Module):
    """ILN (Instance Layer Normalization)"""
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()

        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))

        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        in_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        in_var = torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        ln_mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)

        out = out_in * self.rho.expand(x.shape[0], -1, -1, -1) + out_ln * (1 - self.rho.expand(x.shape[0], -1, -1, -1))
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


class AdaILN(nn.Module):
    """AdaILN (Adaptive Instance Layer Normalization)"""
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()

        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        in_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        in_var = torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        ln_mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)

        out = out_in * self.rho.expand(x.shape[0], -1, -1, -1) + out_ln * (1 - self.rho.expand(x.shape[0], -1, -1, -1))
        out = out * gamma.unsqueeze(dim=2).unsqueeze(dim=3) + beta.unsqueeze(dim=2).unsqueeze(dim=3)
        return out


class ResNetBlock(nn.Module):
    """Residual Block"""
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(n_features)
        ]

        self.main = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.main(x)
        return out


class ResNetAdaLINBlock(nn.Module):
    """ResNet Adaptive Layer Instance Normalization Block"""
    def __init__(self, n_features):
        super(ResNetAdaLINBlock, self).__init__()

        self.pad_conv = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=0, bias=False)
        )
        self.norm = AdaILN(n_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gamma, beta):
        out = self.pad_conv(x)
        out = self.norm(out, gamma, beta)
        out = self.relu(out)
        out = self.pad_conv(out)
        out = self.norm(out, gamma, beta)
        out = out + x
        return out


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self, image_size, num_blocks):
        super(Generator, self).__init__()

        self.in_channels = 3
        self.dim = 64
        self.out_channels = 3
        self.num_blocks = num_blocks
        self.image_size = image_size

        # Down-Sampling #
        down_sampling = []
        down_sampling += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.in_channels, self.dim, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(self.dim),
            nn.ReLU(inplace=True)
        ]

        num_down_sampling = 2
        for i in range(num_down_sampling):
            factor = 2 ** i
            down_sampling += [
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(self.dim * factor, self.dim * factor * 2, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(self.dim * factor * 2),
                nn.ReLU(inplace=True),
            ]

        factor = 2 ** num_down_sampling
        for i in range(num_blocks):
            down_sampling += [
                ResNetBlock(self.dim * factor)
            ]

        self.down_sampling = nn.Sequential(*down_sampling)

        # Class Activation Map (CAM) #
        self.gap_fc = nn.Linear(self.dim * factor, 1, bias=False)
        self.gmp_fc = nn.Linear(self.dim * factor, 1, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(self.dim * factor * 2, self.dim * factor, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        # Block for Gamma and Beta #
        self.fc = nn.Sequential(
            nn.Linear(image_size // factor * image_size // factor * self.dim * factor, self.dim * factor, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * factor, self.dim * factor, bias=False),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Linear(self.dim * factor, self.dim * factor, bias=False)
        self.beta = nn.Linear(self.dim * factor, self.dim * factor, bias=False)

        # Up-Sampling #
        for i in range(num_blocks):
            setattr(self, "UpBlock" + str(i + 1), ResNetAdaLINBlock(self.dim * factor))

        up_sampling = []
        num_up_sampling = 2

        for i in range(num_up_sampling):
            factor = 2 ** (num_up_sampling - i)
            up_sampling += [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(self.dim * factor, int(self.dim * factor / 2), kernel_size=3, stride=1, padding=0, bias=False),
                ILN(int(self.dim * factor / 2)),
                nn.ReLU(inplace=True)
            ]

        up_sampling += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.dim, self.out_channels, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        ]

        self.up_sampling = nn.Sequential(*up_sampling)

    def forward(self, x):
        x = self.down_sampling(x)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logits = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(dim=2).unsqueeze(dim=3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logits = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(dim=2).unsqueeze(dim=3)

        cam_logit = torch.cat([gap_logits, gmp_logits], dim=1)
        x = torch.cat([gap, gmp], dim=1)
        x = self.conv(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)

        out = self.fc(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(out), self.beta(out)

        for i in range(self.num_blocks):
            x = getattr(self, "UpBlock" + str(i + 1))(x, gamma, beta)
        out = self.up_sampling(x)

        return out, cam_logit, heatmap