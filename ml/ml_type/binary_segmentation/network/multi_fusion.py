import os

import torch
from torch.nn import functional as F
from ml.ml_type.base.base_network.base_pt_network import BaseNetwork
from ml.commons.encoder import HighResolution
from ml.commons.decoder import ReFineNetLite
from torch import nn


class BinaryMultiFusion(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = HighResolution(**kwargs)
        layers_feature = [18, 36, 72, 144]
        self.decoder = ReFineNetLite(layers_feature)
        self.final = nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=True)

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        x_out_list = self.encoder(x)

        decoder_output = self.decoder(x_out_list)
        final = F.interpolate(
            self.final(decoder_output), scale_factor=4, mode="bilinear"
        )
        x_out = final

        return x_out
