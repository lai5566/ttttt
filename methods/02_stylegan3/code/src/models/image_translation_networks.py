# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/image_translation_networks.py
# Shared network components for CycleGAN and Pix2Pix

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.ops as ops


class ResnetBlock(nn.Module):
    """ResNet block with reflection padding and optional dropout

    Structure: Conv -> Norm -> ReLU -> Dropout -> Conv -> Norm -> Skip Connection
    """
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward with skip connection"""
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    """ResNet-based generator with configurable number of blocks

    Used in CycleGAN. Architecture:
    - Reflection padding + conv + norm + ReLU
    - 2 downsampling blocks
    - n_blocks ResNet blocks
    - 2 upsampling blocks
    - Output conv + Tanh

    Args:
        input_nc: Number of input channels
        output_nc: Number of output channels
        ngf: Number of generator filters in first conv layer
        norm_layer: Normalization layer type
        use_dropout: Use dropout in ResNet blocks
        n_blocks: Number of ResNet blocks (typically 6 or 9)
        padding_type: Type of padding (reflect, replicate, zero)
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        # Output layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines a U-Net submodule with skip connection

    Structure: downconv -> [submodule] -> upconv
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Add skip connection
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """U-Net generator with skip connections

    Used in Pix2Pix. Architecture builds nested U-Net recursively.

    Args:
        input_nc: Number of input channels
        output_nc: Number of output channels
        num_downs: Number of downsamplings (typically 8 for 256x256)
        ngf: Number of filters in first conv layer
        norm_layer: Normalization layer type
        use_dropout: Use dropout in inner layers
    """
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # Construct U-Net structure recursively
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)

        # Gradually reduce number of filters
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (NLayerDiscriminator)

    Classifies overlapping image patches (receptive field of 70x70 for n_layers=3).
    Can be used for both CycleGAN and Pix2Pix.

    Args:
        input_nc: Number of input channels
        ndf: Number of filters in first conv layer
        n_layers: Number of conv layers (affects receptive field size)
        norm_layer: Normalization layer type
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(PatchGANDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward - returns prediction map, not single scalar"""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """1x1 PatchGAN discriminator (PixelGAN)

    Classifies individual pixels. Very lightweight discriminator.
    Receptive field is 1x1.

    Args:
        input_nc: Number of input channels
        ndf: Number of filters in first layer
        norm_layer: Normalization layer type
    """
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.InstanceNorm2d):
        super(PixelDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward"""
        return self.net(input)


# Import functools for partial (needed by norm_layer checks)
import functools


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Args:
        norm_type (str): name of normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters and do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights

    Args:
        net (network): network to be initialized
        init_type (str): name of initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def define_generator(gen_type='resnet_9blocks', input_nc=3, output_nc=3, ngf=64,
                    norm='instance', use_dropout=False, init_type='normal', init_gain=0.02):
    """Create and initialize a generator

    Args:
        gen_type (str): type of generator architecture: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        input_nc (int): number of input image channels
        output_nc (int): number of output image channels
        ngf (int): number of filters in the last conv layer
        norm (str): type of normalization layer: batch | instance | none
        use_dropout (bool): if use dropout layers
        init_type (str): name of initialization method
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        Generator network
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if gen_type == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                             use_dropout=use_dropout, n_blocks=9)
    elif gen_type == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                             use_dropout=use_dropout, n_blocks=6)
    elif gen_type == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)
    elif gen_type == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout)
    else:
        raise NotImplementedError(f'Generator type [{gen_type}] is not recognized')

    init_weights(net, init_type, init_gain)
    return net


def define_discriminator(disc_type='basic', input_nc=3, ndf=64, n_layers_D=3,
                        norm='instance', init_type='normal', init_gain=0.02):
    """Create and initialize a discriminator

    Args:
        disc_type (str): type of discriminator architecture: basic | n_layers | pixel
        input_nc (int): number of input image channels
        ndf (int): number of filters in the first conv layer
        n_layers_D (int): only used if disc_type=='n_layers'
        norm (str): type of normalization layer: batch | instance | none
        init_type (str): name of initialization method
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        Discriminator network
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if disc_type == 'basic':
        net = PatchGANDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif disc_type == 'n_layers':
        net = PatchGANDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    elif disc_type == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f'Discriminator type [{disc_type}] is not recognized')

    init_weights(net, init_type, init_gain)
    return net
