from torch import nn
from torch.nn import functional as F


class BiLinearUpSampling(nn.Module):
    def __init__(self, down_factor, in_features, num_classes):
        super().__init__()
        self.down_factor = down_factor
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.down_factor, mode="bilinear", align_corners=True
        )
