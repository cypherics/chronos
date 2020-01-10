import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__reference__ = [
    "https://github.com/bfortuner/pytorch_tiramisu/blob/master/models/",
    "https://gitlab.com/theICTlab/UrbanReconstruction/ictnet/blob/master/code/",
]
__all__ = ["DenseNet"]


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)]


class SpatialAttentionFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, low_level_features, high_level_features):
        """

        :param low_level_features: ict_net extracted from backbone
        :param high_level_features: up sampled ict_net
        :return:
        """
        high_level_features_sigmoid = high_level_features.sigmoid()
        weighted_low_level_features = high_level_features_sigmoid * low_level_features

        feature_fusion = weighted_low_level_features + high_level_features
        return feature_fusion


class SqueezeExcitation(nn.Module):
    def forward(self, x, shape, ratio, out_filter):
        _, model_filter, w, h = shape

        if out_filter is None:
            out_filter = model_filter

        global_avg_pool = F.avg_pool2d(x, kernel_size=(w, h), stride=1).view(
            x.shape[:-2], -1
        )
        # global_avg_pool = nn.AdaptiveAvgPool2d(1)(x)
        dense_layer_1 = nn.Linear(model_filter, out_filter).cuda()(global_avg_pool)
        non_linearity = nn.ReLU(inplace=True)(dense_layer_1)

        dense_layer_2 = nn.Linear(out_filter, out_filter).cuda()(non_linearity)
        sigmoid_non_linearity = dense_layer_2.sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * sigmoid_non_linearity


class _DenseLayer(nn.Sequential):
    """
    Code for ict_net borrowed from https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    """

    def __init__(self, num_input_features, growth_rate, drop_rate, batch_momentum):
        super(_DenseLayer, self).__init__()
        self.add_module(
            "norm1",
            nn.BatchNorm2d(
                num_input_features, momentum=batch_momentum, track_running_stats=True
            ),
        ),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.Module):
    """
    Code for ict_net borrowed from https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    """

    def __init__(
        self,
        num_layers,
        num_input_features,
        growth_rate,
        drop_rate,
        out_filter=None,
        down_sample=True,
        batch_momentum=0.1,
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
        self.out_filter = out_filter
        self.down_sample = down_sample

    def forward(self, x):
        new_feat = list()
        for layer in self.dense_module:
            dense_layer_output = layer(x)
            new_feat.append(dense_layer_output)
            x = torch.cat([x, dense_layer_output], 1)

        if self.down_sample:
            return SqueezeExcitation().forward(
                x, x.shape, 1, out_filter=self.out_filter
            )
        else:
            x = torch.cat(new_feat, 1)
            return SqueezeExcitation().forward(
                x, x.shape, 1, out_filter=self.out_filter
            )


class _TransitionDown(nn.Sequential):
    """
    Code for ict_net borrowed from https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    """

    def __init__(
        self, num_input_features, num_output_features, drop_rate, batch_momentum
    ):
        super(_TransitionDown, self).__init__()
        self.add_module(
            "norm",
            nn.BatchNorm2d(
                num_input_features, momentum=batch_momentum, track_running_stats=True
            ),
        )
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        if drop_rate > 0:
            self.add_module("dropout", nn.Dropout(p=drop_rate))
        self.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2))


class _TransitionUp(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.Transpose = nn.ConvTranspose2d(
            in_channels=num_input_features,
            out_channels=num_output_features,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x, skip):
        out = self.Transpose(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class DenseNet(nn.Module):
    """
    Code for ict_net borrowed from https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    """

    def __init__(
        self,
        growth_rate,
        encoder_block_config,
        decoder_block_config,
        bottleneck_layers,
        num_init_features=48,
        drop_rate=0.0,
        batch_momentum=0.01,
    ):

        super(DenseNet, self).__init__()
        self.decoder = nn.ModuleList()
        skip_connection_channel_counts = []
        # First convolution

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    ("relu0", nn.ReLU(inplace=True)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(encoder_block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                batch_momentum=batch_momentum,
            )
            self.encoder.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            # In original densenet number of features are divided by 2 for _TransitionDown
            trans = _TransitionDown(
                num_input_features=num_features,
                num_output_features=num_features,
                drop_rate=drop_rate,
                batch_momentum=batch_momentum,
            )
            self.encoder.add_module("transition%d" % (i + 1), trans)
            skip_connection_channel_counts.insert(0, num_features)
        self.bottle_neck = DenseNetBottleNeck(
            num_features, growth_rate, bottleneck_layers
        )

        num_features = growth_rate * bottleneck_layers
        for j, num_layers in enumerate(decoder_block_config):
            trans = _TransitionUp(num_features, num_features)
            self.decoder.add_module("decodertransition%d" % (j + 1), trans)

            cur_channels_count = num_features + skip_connection_channel_counts[j]
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=cur_channels_count,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                down_sample=False,
            )
            self.decoder.add_module("decoderdenseblock%d" % (j + 1), block)
            num_features = growth_rate * decoder_block_config[j]

    def forward(self, x):
        encoder_features = self.encoder(x)
        bottle_neck = self.bottle_neck(encoder_features)
        decoder_features = self.decoder(bottle_neck)
        return encoder_features, bottle_neck, decoder_features


class DenseNetBottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.bottle_neck = _DenseBlock(
            growth_rate=growth_rate,
            num_layers=num_layers,
            drop_rate=0.2,
            num_input_features=in_channels,
            out_filter=growth_rate * num_layers,
            down_sample=False,
        )

    def forward(self, x):
        return self.bottle_neck(x)


class ICTNet(nn.Module):
    def __init__(
        self,
        classes=2,
        drop_rate=0.2,
        batch_momentum=0.1,
        growth_rate=12,
        layers_per_block=4,
    ):
        super().__init__()
        if isinstance(layers_per_block, int):
            per_block = [layers_per_block] * 11
        elif isinstance(layers_per_block, list):
            assert len(layers_per_block) == 11
            per_block = layers_per_block
        else:
            raise ValueError

        if growth_rate == 12 and layers_per_block == 4:
            final_layer_features = 48
        elif growth_rate == 16 and isinstance(layers_per_block, list):
            final_layer_features = 64
        else:
            raise ValueError

        self.ict_net = DenseNet(
            drop_rate=drop_rate,
            batch_momentum=batch_momentum,
            growth_rate=growth_rate,
            encoder_block_config=per_block[0:5],
            decoder_block_config=per_block[6:11],
            bottleneck_layers=per_block[5:6][0],
        )
        self.final_layer = nn.Conv2d(
            final_layer_features, classes, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, image_dict):
        if isinstance(image_dict, dict):
            x = image_dict["image"]
        else:
            x = image_dict
        skip_connections = list()
        dense_layer_0 = nn.Sequential(
            self.ict_net.encoder.conv0, self.ict_net.encoder.relu0
        )(x)

        dense_layer_1 = self.ict_net.encoder.denseblock1(dense_layer_0)
        skip_connections.append(dense_layer_1)
        transition_down_1 = self.ict_net.encoder.transition1(dense_layer_1)

        dense_layer_2 = self.ict_net.encoder.denseblock2(transition_down_1)
        skip_connections.append(dense_layer_2)
        transition_down_2 = self.ict_net.encoder.transition2(dense_layer_2)

        dense_layer_3 = self.ict_net.encoder.denseblock3(transition_down_2)
        skip_connections.append(dense_layer_3)
        transition_down_3 = self.ict_net.encoder.transition3(dense_layer_3)

        dense_layer_4 = self.ict_net.encoder.denseblock4(transition_down_3)
        skip_connections.append(dense_layer_4)
        transition_down_4 = self.ict_net.encoder.transition4(dense_layer_4)

        dense_layer_5 = self.ict_net.encoder.denseblock5(transition_down_4)
        skip_connections.append(dense_layer_5)
        transition_down_5 = self.ict_net.encoder.transition5(dense_layer_5)

        bottle_neck = self.ict_net.bottle_neck(transition_down_5)
        transition_up_1 = self.ict_net.decoder.decodertransition1(
            bottle_neck, skip_connections.pop()
        )
        dense_layer_6 = self.ict_net.decoder.decoderdenseblock1(transition_up_1)

        transition_up_2 = self.ict_net.decoder.decodertransition2(
            dense_layer_6, skip_connections.pop()
        )
        dense_layer_7 = self.ict_net.decoder.decoderdenseblock2(transition_up_2)

        transition_up_3 = self.ict_net.decoder.decodertransition3(
            dense_layer_7, skip_connections.pop()
        )
        dense_layer_8 = self.ict_net.decoder.decoderdenseblock3(transition_up_3)

        transition_up_4 = self.ict_net.decoder.decodertransition4(
            dense_layer_8, skip_connections.pop()
        )
        dense_layer_9 = self.ict_net.decoder.decoderdenseblock4(transition_up_4)

        transition_up_5 = self.ict_net.decoder.decodertransition5(
            dense_layer_9, skip_connections.pop()
        )
        dense_layer_10 = self.ict_net.decoder.decoderdenseblock5(transition_up_5)

        final_layer = self.final_layer(dense_layer_10)
        return final_layer


# if __name__ == "__main__":
#     model = ICTNet()
#     model.eval()
#     image = torch.randn(2, 3, 224, 224)
#     image_temp = torch.randn(2, 3, 288, 288)
#     model.cuda()
#     from utility.torch_tensor_conversion import cuda_variable
#
#     with torch.no_grad():
#         output = model.forward({"image": cuda_variable(image)})
#     a = tuple(output.shape)
#     print(a)
