import torch
from torch import nn
from torch.nn import functional as F


class SpatialPooling(nn.Module):
    def __init__(self):
        super(SpatialPooling, self).__init__()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(5, 5))
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(6, 6))

    def forward(self, x):
        _, _, w, h = x.shape
        p1 = F.upsample(self.max_pool_1(x), size=(w, h), mode="bilinear")
        p2 = F.upsample(self.max_pool_2(x), size=(w, h), mode="bilinear")
        p3 = F.upsample(self.max_pool_3(x), size=(w, h), mode="bilinear")
        p4 = F.upsample(self.max_pool_4(x), size=(w, h), mode="bilinear")

        out = torch.cat([p1, p2, p3, p4, x], 1)
        return out
