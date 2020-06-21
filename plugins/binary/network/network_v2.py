from torch.nn import functional as F
from plugins.base.network.base_network import BaseNetwork
from ml.modules.encoder import HighResolution
from ml.modules.decoder import ReFineNetLite
from torch import nn


class BinaryRefineLite(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = HighResolution(**kwargs)
        layers_feature = [18, 36, 72, 144]
        self.decoder = ReFineNetLite(layers_feature)

        self.final = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        x_out_list = self.encoder(x)

        decoder_output = self.decoder(x_out_list)
        final = F.interpolate(
            self.final(decoder_output), scale_factor=4, mode="bilinear"
        )

        x_out = final

        return x_out


# if __name__ == "__main__":
#     import torch
#
#     model = BinaryRefineLite(
#         **{"pre_trained_pascal": "/home/palnak/Downloads/hrnetv2_w18_imagenet_pretrained.pth",
#             "STAGE1": {
#                 "NUM_CHANNELS": 64,
#                 "BLOCK": "BOTTLENECK",
#                 "NUM_BLOCKS": 4,
#                 "FUSE_METHOD": "SUM",
#                 "NUM_MODULES": 1,
#                 "NUM_BRANCHES": 1,
#             },
#             "STAGE2": {
#                 "NUM_CHANNELS": [18, 36],
#                 "BLOCK": "BASIC",
#                 "NUM_BLOCKS": [4, 4],
#                 "FUSE_METHOD": "SUM",
#                 "NUM_MODULES": 1,
#                 "NUM_BRANCHES": 2,
#             },
#             "STAGE3": {
#                 "NUM_CHANNELS": [18, 36, 72],
#                 "BLOCK": "BASIC",
#                 "NUM_BLOCKS": [4, 4, 4],
#                 "FUSE_METHOD": "SUM",
#                 "NUM_MODULES": 4,
#                 "NUM_BRANCHES": 3,
#             },
#             "STAGE4": {
#                 "NUM_CHANNELS": [18, 36, 72, 144],
#                 "BLOCK": "BASIC",
#                 "NUM_BLOCKS": [4, 4, 4, 4],
#                 "FUSE_METHOD": "SUM",
#                 "NUM_MODULES": 3,
#                 "NUM_BRANCHES": 4,
#             }
#         }
#     )
#     model.eval()
#     image = torch.randn(1, 3, 384, 384)
#     with torch.no_grad():
#         output = model.forward_propagate({"image": image})
#     a = tuple(output.shape)
#     print(a)
