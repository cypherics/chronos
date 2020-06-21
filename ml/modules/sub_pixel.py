from torch import nn


class SubPixel(nn.Module):
    def __init__(self, down_factor, in_features, num_classes):
        super(SubPixel, self).__init__()
        features = (down_factor ** 2) * num_classes
        self.convolution = nn.Conv2d(in_features, features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(features)
        self.non_linearity = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.convolution(x)
        x = self.bn(x)
        x = self.non_linearity(x)
        x = self.pixel_shuffle(x)
        return x
