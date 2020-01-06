import math

from torch import nn
import torch


def icnr(x, scale=2, init=nn.init.kaiming_normal):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    sub_kernel = torch.zeros(new_shape)
    sub_kernel = init(sub_kernel)
    sub_kernel = sub_kernel.transpose(0, 1)
    sub_kernel = sub_kernel.contiguous().view(
        sub_kernel.shape[0], sub_kernel.shape[1], -1
    )

    kernel = sub_kernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


def conv(ni, nf, kernel_size=3, actn=False):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2)]
    if actn:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.m(x) * self.res_scale


def res_block(nf):
    return ResSequential([conv(nf, nf, actn=True), conv(nf, nf)], 0.1)


def pixel_up_sample(ni, nf, scale):
    layers = list()
    for i in range(int(math.log(scale, 2))):
        layers += [conv(ni, nf * 4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


class SrResNet(nn.Module):
    def __init__(self, nf=64, scale=4):
        super().__init__()
        features = [conv(3, 64)]
        for i in range(8):
            features.append(res_block(64))
        features += [
            conv(64, 64),
            pixel_up_sample(64, 64, scale),
            nn.BatchNorm2d(64),
            conv(64, 3),
        ]
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


class HighResolutionModel(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.sr_resnet_model = SrResNet(64, scale)
        self.weight_initialization(scale)

    def weight_initialization(self, scale):
        conv_shuffle = self.sr_resnet_model.features[10][0][0]
        kernel = icnr(conv_shuffle.weight, scale=scale)
        conv_shuffle.weight.data.copy_(kernel)

        conv_shuffle = self.sr_resnet_model.features[10][2][0]
        kernel = icnr(conv_shuffle.weight, scale=scale)
        conv_shuffle.weight.data.copy_(kernel)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["image"]
        return self.sr_resnet_model(x)


# def sr_resnet():
#     sr_resnet_model = SrResnet(64, 4)
#     conv_shuffle = sr_resnet_model.features[10][0][0]
#     kernel = icnr(conv_shuffle.weight, scale=2)
#     conv_shuffle.weight.data.copy_(kernel)
#
#     conv_shuffle = sr_resnet_model.features[10][2][0]
#     kernel = icnr(conv_shuffle.weight, scale=4)
#     conv_shuffle.weight.data.copy_(kernel)
#     return sr_resnet_model
