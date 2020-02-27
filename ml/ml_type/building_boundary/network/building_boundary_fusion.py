from torch.nn import functional as F
from ml.ml_type.base.base_network.base_pt_network import BaseNetwork
from ml.commons.encoder import HighResolution
from ml.commons.decoder import DecoderDenseNet
from torch import nn


class Convolution3x3BNAct(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU()
        self.convolution = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BuildingBoundaryExtractor(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = HighResolution(**kwargs)
        layers_feature = [18, 36, 72, 144]
        self.decoder = DecoderDenseNet(
            growth_rate=12,
            decoder_block_config=[4, 4, 4, 4],
            bottleneck_layers=4,
            encoder_layers_features=layers_feature,
            compression_ratio=1
        )

        self.convolution_1 = Convolution3x3BNAct(
            in_channel=self.decoder.last_layer_feature, out_channel=32
        )
        self.convolution_2 = Convolution3x3BNAct(in_channel=32, out_channel=32)

        self.final = nn.Conv2d(32, 2, kernel_size=1)

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        x_out_list = self.encoder(x)

        decoder_output = self.decoder(x_out_list)
        # final = F.interpolate(
        #     self.final(decoder_output), scale_factor=4, mode="bilinear"
        # )
        inter_1 = F.interpolate(decoder_output, scale_factor=2, mode="nearest")
        layer_1 = self.convolution_1(inter_1)

        inter_2 = F.interpolate(layer_1, scale_factor=2, mode="nearest")
        layer_2 = self.convolution_2(inter_2)

        final = self.final(layer_2)
        x_out = final

        return x_out
