from torch import nn

from ml.ml_type.base.base_network.base_pt_network import BaseNetwork
from torchvision import models
from torch.nn import functional as F

from ml.commons.decoder import ReFineNetLite


class BinaryReFineLiteExtractor(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if hasattr(models, kwargs["backbone_to_use"]):
            back_bone = getattr(models, kwargs["backbone_to_use"])(
                pretrained=kwargs["pre_trained_image_net"]
            )

            self.layer0 = nn.Sequential(
                back_bone.conv1, back_bone.bn1, back_bone.relu, back_bone.maxpool
            )
            self.layer1 = back_bone.layer1
            self.layer2 = back_bone.layer2
            self.layer3 = back_bone.layer3
            self.layer4 = back_bone.layer4

        layers_feature = [64, 128, 256, 512]
        self.decoder = ReFineNetLite(layers_feature)
        self.final = nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=True)

    def forward_propagate(self, input_feature):
        x = input_feature["image"]

        x_out_list = list()
        layer_0_output = self.layer0(x)
        layer_1_output = self.layer1(layer_0_output)  # 1/4
        x_out_list.append(layer_1_output)

        layer_2_output = self.layer2(layer_1_output)  # 1/8
        x_out_list.append(layer_2_output)

        layer_3_output = self.layer3(layer_2_output)  # 1/16
        x_out_list.append(layer_3_output)

        layer_4_output = self.layer4(layer_3_output)  # 1/32
        x_out_list.append(layer_4_output)

        decoder_output = self.decoder(x_out_list)
        final = F.interpolate(
            self.final(decoder_output), scale_factor=4, mode="bilinear"
        )
        x_out = final

        return x_out

