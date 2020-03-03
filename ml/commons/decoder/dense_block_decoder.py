import torch
from torch import nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    """
    Code  borrowed from https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    """

    def __init__(self, num_input_features, growth_rate, drop_rate, batch_momentum):
        super(_DenseLayer, self).__init__()
        # self.add_module(
        #     "norm1", nn.BatchNorm2d(num_input_features, momentum=batch_momentum)
        # ),
        # self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        ),
        self.add_module("norm1", nn.BatchNorm2d(growth_rate, momentum=batch_momentum))
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        num_input_features,
        growth_rate,
        drop_rate,
        batch_momentum=0.01,
    ):
        super(_DenseBlock, self).__init__()
        self.dense_module = nn.ModuleList(
            [
                _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate,
                    drop_rate,
                    batch_momentum=batch_momentum,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        new_feat = list()
        for layer in self.dense_module:
            dense_layer_output = layer(x)
            new_feat.append(dense_layer_output)
            x = torch.cat([x, dense_layer_output], 1)
        return x


class _TransitionUp(nn.Module):
    def __init__(self, num_input_features, compression_ratio=0.5):
        super().__init__()
        num_output_features = int(num_input_features * compression_ratio)
        self.Transpose = nn.ConvTranspose2d(
            in_channels=num_input_features,
            out_channels=num_output_features,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x, skip):
        out = self.Transpose(x)
        out = torch.cat([out, skip], 1)
        return out


class DenseNetBottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.bottle_neck = _DenseBlock(
            growth_rate=growth_rate,
            num_layers=num_layers,
            drop_rate=0.2,
            num_input_features=in_channels,
        )

    def forward(self, x):
        return self.bottle_neck(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate,
        decoder_block_config,
        bottleneck_layers,
        num_init_features,
        drop_rate,
        compression_ratio,
        skip_connection_channel_counts,
    ):
        super(DenseNet, self).__init__()
        self.decoder = nn.ModuleList()
        self.final_layer_feature = None

        self.trans = _TransitionDown(
            num_input_features=num_init_features, num_output_features=num_init_features
        )
        self.bottle_neck = DenseNetBottleNeck(
            num_init_features, growth_rate, bottleneck_layers
        )

        num_features = num_init_features + growth_rate * bottleneck_layers
        for j, num_layers in enumerate(decoder_block_config):
            trans = _TransitionUp(num_features, compression_ratio)
            self.decoder.add_module("decodertransition%d" % (j + 1), trans)

            trans_out_feature = int(num_features * compression_ratio)
            cur_channels_count = trans_out_feature + skip_connection_channel_counts[j]
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=cur_channels_count,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.decoder.add_module("decoderdenseblock%d" % (j + 1), block)
            num_features = cur_channels_count + growth_rate * decoder_block_config[j]
        self.final_layer_feature = num_features

    def forward(self, x):
        pass


class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionDown, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DecoderDenseNet(nn.Module):
    def __init__(
        self,
        growth_rate,
        decoder_block_config,
        bottleneck_layers,
        encoder_layers_features,
        drop_rate=0.0,
        compression_ratio=1,
    ):
        super(DecoderDenseNet, self).__init__()
        num_init_features = encoder_layers_features[-1]
        skip_connection_channel_counts = [
            encoder_layers_features[-1],
            encoder_layers_features[-2],
            encoder_layers_features[-3],
            encoder_layers_features[-4],
        ]
        self.dense_decoder = DenseNet(
            growth_rate,
            decoder_block_config,
            bottleneck_layers,
            num_init_features,
            drop_rate,
            compression_ratio,
            skip_connection_channel_counts,
        )
        self.last_layer_feature = self.dense_decoder.final_layer_feature

    def forward(self, encoder_features):

        bottle_neck = self.dense_decoder.bottle_neck(
            self.dense_decoder.trans(encoder_features[-1])
        )

        transition_up_1 = self.dense_decoder.decoder.decodertransition1(
            bottle_neck, encoder_features[-1]
        )
        dense_layer_6 = self.dense_decoder.decoder.decoderdenseblock1(transition_up_1)

        transition_up_2 = self.dense_decoder.decoder.decodertransition2(
            dense_layer_6, encoder_features[-2]
        )
        dense_layer_7 = self.dense_decoder.decoder.decoderdenseblock2(transition_up_2)

        transition_up_3 = self.dense_decoder.decoder.decodertransition3(
            dense_layer_7, encoder_features[-3]
        )
        dense_layer_8 = self.dense_decoder.decoder.decoderdenseblock3(transition_up_3)

        transition_up_4 = self.dense_decoder.decoder.decodertransition4(
            dense_layer_8, encoder_features[-4]
        )
        dense_layer_9 = self.dense_decoder.decoder.decoderdenseblock4(transition_up_4)

        return dense_layer_9
