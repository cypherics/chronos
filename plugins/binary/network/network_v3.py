from torch.nn import functional as F
from plugins.base.network.base_network import BaseNetwork
from ml.modules.decoder import ReFineNetLite
from torch import nn
from torchvision.models.resnet import resnet34


class BinaryRefineLiteV1(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = resnet34(pretrained=True)
        self.layer0 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
        )
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        layers_feature = [64, 128, 256, 512]
        self.decoder = ReFineNetLite(layers_feature)

        self.final = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        layer_0_output = self.layer0(x)
        layer_1_output = self.layer1(layer_0_output)  # 1/4
        layer_2_output = self.layer2(layer_1_output)  # 1/8
        layer_3_output = self.layer3(layer_2_output)  # 1/16
        layer_4_output = self.layer4(layer_3_output)  # 1/32

        x_out_list = [layer_1_output, layer_2_output, layer_3_output, layer_4_output]

        decoder_output = self.decoder(x_out_list)
        final = F.interpolate(
            self.final(decoder_output), scale_factor=4, mode="bilinear"
        )

        x_out = final

        return x_out


#
# if __name__ == "__main__":
#     import torch
#
#     model = BinaryRefineLite()
#     model.eval()
#     image = torch.randn(1, 3, 384, 384)
#     with torch.no_grad():
#         output = model.forward_propagate({"image": image})
#     a = tuple(output.shape)
#     print(a)
