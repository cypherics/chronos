from torch import nn
from torchvision import models

from ml.base.base_network.base_pt_network import BaseNetwork
from ml.commons.layers import UpSampleConvolution
from ml.commons.decoder import ReFineNet


def convolution_3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


class BuildingBoundaryRefineNetExtractor(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        backbone_to_use = kwargs["backbone_to_use"]

        if hasattr(models, backbone_to_use):
            back_bone = getattr(models, backbone_to_use)(
                pretrained=kwargs["pre_trained_image_net"]
            )

            self.layer0 = nn.Sequential(
                back_bone.conv1, back_bone.bn1, back_bone.relu, back_bone.maxpool
            )
            self.layer1 = back_bone.layer1
            self.layer2 = back_bone.layer2
            self.layer3 = back_bone.layer3
            self.layer4 = back_bone.layer4
        else:
            raise ModuleNotFoundError

        if backbone_to_use == "resnet50":
            layers_features = [256, 512, 1024, 2048]

        elif backbone_to_use == "resnet34":
            layers_features = [64, 128, 256, 512]

        else:
            raise NotImplementedError

        self.decoder_model = ReFineNet(layers_features)
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
        layer_0_output = self.layer0(x)
        layer_1_output = self.layer1(layer_0_output)  # 1/4
        layer_2_output = self.layer2(layer_1_output)  # 1/8
        layer_3_output = self.layer3(layer_2_output)  # 1/16
        layer_4_output = self.layer4(layer_3_output)  # 1/32

        x_out = self.decoder_model(
            [layer_1_output, layer_2_output, layer_3_output, layer_4_output]
        )

        if self.final_mode == "bilinear":
            final_map = self.final_layer(x_out)
            final_map = self.final_layer_up_sample(final_map)
        else:
            final_map = self.final_layer_up_sample(x_out)

        return final_map
