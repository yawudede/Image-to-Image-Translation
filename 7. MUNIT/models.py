import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, n_features, norm='instance'):
        super(ResidualBlock, self).__init__()

        if norm == 'instance':
            normalization_layer = nn.InstanceNorm2d
        elif norm == 'adain':
            normalization_layer = AdaptiveInstanceNorm2d
        else:
            raise NotImplementedError

        conv_block = [
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=0, bias=True),
            normalization_layer(n_features),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d(padding=1),
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=0, bias=True),
            normalization_layer(n_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ContentEncoder(nn.Module):
    """Content Encoder Network"""
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.in_channels = 3
        self.dim = 64
        self.num_downsamples = 2
        self.num_resblocks = 3

        layers = []
        layers += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.in_channels, self.dim, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(self.dim),
            nn.ReLU(inplace=True)
        ]

        for i in range(self.num_downsamples):
            mult = 2 ** i

            layers += [
                nn.Conv2d(self.dim * mult, self.dim * 2 * mult, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(self.dim * 2),
                nn.ReLU(inplace=True)
            ]

        mult = 2 ** self.num_downsamples

        for j in range(self.num_resblocks):
            layers += [
                ResidualBlock(self.dim * mult)
            ]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class StyleEncoder(nn.Module):
    """Style Encoder Network"""
    def __init__(self, style_dim=8):
        super(StyleEncoder, self).__init__()

        self.in_channels = 3
        self.dim = 64
        self.num_downsamples = 2
        self.style_dim = style_dim

        layers = []
        layers += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(self.in_channels, self.dim, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True)
        ]

        for i in range(self.num_downsamples):
            mult = 2 ** i

            layers += [
                nn.Conv2d(self.dim * mult, self.dim * 2 * mult, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]

        mult = 2 ** self.num_downsamples

        layers += [
            nn.Conv2d(self.dim * mult, self.dim * mult, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(self.dim * mult, style_dim, kernel_size=1, stride=1, padding=0)
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class Decoder(nn.Module):
    """Decoder Network"""
    def __init__(self, style_dim=8):
        super(Decoder, self).__init__()

        self.num_resblocks = 4
        self.num_upsamples = 2
        self.dim = 64
        self.out_channel = 3

        layers = []
        dim = self.dim * (2 ** self.num_upsamples)

        # Residual Blocks #
        for i in range(self.num_resblocks):
            layers += [
                ResidualBlock(dim, norm='adain')
            ]

        # Up-Sampling #
        for j in range(self.num_upsamples):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True)
            ]
            dim = dim // 2

        # Output Layer #
        layers += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(dim, self.out_channel, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class MLP(nn.Module):
    """Multi-Layer Perceptron for Predicting AdaIN Parameters"""
    def __init__(self, in_dim, dim, out_dim):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.dim = 256

        layers = []
        layers += [
            nn.Linear(in_dim, self.dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.dim, out_dim)
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.main(x)
        return out


class Multi_Discriminator(nn.Module):
    """Multiple Discriminator Network"""
    def __init__(self, in_channels=3):
        super(Multi_Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1
        self.num_disc = 3

        self.models = nn.ModuleList()
        for i in range(self.num_disc):
            self.models.add_module('disc_%d' % i,
                                   nn.Sequential(
                                       nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2, padding=1,
                                                 bias=True),
                                       nn.LeakyReLU(0.2, inplace=True),

                                       nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=True),
                                       nn.LeakyReLU(0.2, inplace=True),

                                       nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1,
                                                 bias=True),
                                       nn.LeakyReLU(0.2, inplace=True),

                                       nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1,
                                                 bias=True),
                                       nn.LeakyReLU(0.2, inplace=True),

                                       nn.Conv2d(self.ndf * 8, self.out_channel, kernel_size=1, stride=1, padding=0,
                                                 bias=True)
                                   )
                                   )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class AdaIN_Generator(nn.Module):
    """Adaptive Instance Normalization Generator"""
    def __init__(self, style_dim=8):
        super(AdaIN_Generator, self).__init__()

        self.style_dim = style_dim
        self.mlp_dim = 256

        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.decoder = Decoder()
        self.mlp = MLP(style_dim, self.mlp_dim, self.get_num_adain_params(self.decoder))

    def encode(self, images):
        content = self.content_encoder(images)
        style = self.style_encoder(images)
        return content, style

    def decode(self, content, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, model=self.decoder)
        out = self.decoder(content)
        return out

    def forward(self, images):
        content, style = self.encode(images)
        image = self.decode(content, style)
        return image

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = nn.functional.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
