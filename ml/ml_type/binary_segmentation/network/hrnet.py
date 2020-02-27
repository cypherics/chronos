import os

import torch
from torch.nn import functional as F
from ml.ml_type.base.base_network.base_pt_network import BaseNetwork
from ml.commons.encoder import HighResolution
from torch import nn


class PlainHRNet(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = HighResolution(**kwargs)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.last_inp_channels,
                out_channels=self.encoder.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(self.encoder.last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.encoder.last_inp_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0)
        )

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        x_out_list = self.encoder(x)

        # Upsampling
        x0_h, x0_w = x_out_list[0].size(2), x_out_list[0].size(3)
        x1 = F.upsample(x_out_list[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x_out_list[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x_out_list[3], size=(x0_h, x0_w), mode='bilinear')

        final = torch.cat([x_out_list[0], x1, x2, x3], 1)

        x_out = self.last_layer(final)

        return x_out
