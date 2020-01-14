from torch import nn

from ml.base.base_network.base_pt_network import BaseNetwork
from ml.commons.layers import UpSampleConvolution
from ml.commons.network import ReFineNet


def convolution_3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


class BuildingBoundaryRefineNetExtractor(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = ReFineNet(
            kwargs["backbone_to_use"],
            kwargs["pre_trained_image_net"],
            kwargs["top_layers_trainable"],
        )
        self.num_classes = kwargs["classes"]
        self.final_mode = kwargs["final_mode"]

        self.final_layer = convolution_3x3(in_planes=256, out_planes=self.num_classes)
        self.final_layer_up_sample = UpSampleConvolution().init_method(
            method=self.final_mode,
            down_factor=4,
            in_features=256,
            num_classes=self.num_classes,
        )

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        x_out = self.model(x)

        if self.final_mode == "bilinear":
            final_map = self.final_layer(x_out)
            final_map = self.final_layer_up_sample(final_map)
        else:
            final_map = self.final_layer_up_sample(x_out)

        return final_map
