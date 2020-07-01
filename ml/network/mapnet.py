from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from ml.modules import SpatialPooling, ChannelSELayer
from plugins.base.network.base_network import BaseNetwork

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class Convolution2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super().__init__()
        self.convolution_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.init_weights()

    def forward(self, x):
        x = self.convolution_layer(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BnActCon(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super().__init__()
        self.bn = BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.convolution_layer = Convolution2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.activation = nn.ReLU(inplace=True)
        self.init_weights()

    def forward(self, x):
        x = self.bn(x)
        x = self.activation(x)
        x = self.convolution_layer(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)

        self.bn1 = BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.convolution_layer_1 = Convolution2d(
            in_channels, out_channel, kernel_size=3, padding=1
        )

        self.bn2 = BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.convolution_layer_2 = Convolution2d(
            out_channel, out_channel, kernel_size=3, padding=1
        )

    def forward(self, x):
        residual = x
        x_1 = self.bn1(x)
        x_1 = self.activation(x_1)
        x_1 = self.convolution_layer_1(x_1)

        x_2 = self.bn2(x_1)
        x_2 = self.activation(x_2)
        x_2 = self.convolution_layer_2(x_2)

        out = residual + x_2
        return out


class BottleNeck(nn.Module):
    def __init__(
        self, in_channels, bottle_neck_channel, expansion=4, down_sample=False
    ):
        super().__init__()
        self.down_sample = down_sample
        self.expansion = expansion
        self.activation = nn.ReLU(inplace=True)

        self.bn1 = BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.convolution_layer_1 = Convolution2d(
            in_channels, bottle_neck_channel, kernel_size=1, padding=0
        )

        self.bn2 = BatchNorm2d(bottle_neck_channel, momentum=BN_MOMENTUM)
        self.convolution_layer_2 = Convolution2d(
            bottle_neck_channel, bottle_neck_channel
        )

        self.bn3 = BatchNorm2d(bottle_neck_channel, momentum=BN_MOMENTUM)
        self.convolution_layer_3 = Convolution2d(
            bottle_neck_channel,
            bottle_neck_channel * self.expansion,
            kernel_size=1,
            padding=0,
        )

        if self.down_sample:
            self.bn_down_sample = BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
            self.convolution_layer_down_sample = Convolution2d(
                in_channels,
                bottle_neck_channel * self.expansion,
                kernel_size=1,
                padding=0,
            )

    def forward(self, x):
        residual = x
        x_1 = self.bn1(x)
        x_1 = self.activation(x_1)
        x_1 = self.convolution_layer_1(x_1)

        x_2 = self.bn2(x_1)
        x_2 = self.activation(x_2)
        x_2 = self.convolution_layer_2(x_2)

        x_3 = self.bn3(x_2)
        x_3 = self.activation(x_3)
        x_3 = self.convolution_layer_3(x_3)

        if self.down_sample:
            residual = self.bn_down_sample(x)
            residual = self.activation(residual)
            residual = self.convolution_layer_down_sample(residual)
        out = residual + x_3
        return out


class Step0(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottle_neck_1 = BottleNeck(
            in_channels=64, bottle_neck_channel=64, down_sample=True
        )
        self.bottle_neck_2 = BottleNeck(in_channels=256, bottle_neck_channel=64)
        self.bottle_neck_3 = BottleNeck(in_channels=256, bottle_neck_channel=64)
        self.bottle_neck_4 = BottleNeck(in_channels=256, bottle_neck_channel=64)

    def forward(self, x):
        x = self.bottle_neck_1(x)
        x = self.bottle_neck_2(x)
        x = self.bottle_neck_3(x)
        x = self.bottle_neck_4(x)
        return x


class Stage(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_modules=2,
        block_num=4,
        multi_scale_output=True,
    ):
        super().__init__()
        self.stage_res = list()
        self.fuse_res = OrderedDict()
        fuse_index = 0

        self.block_num = block_num
        self.num_modules = num_modules
        self.multi_scale_output = multi_scale_output
        for i in range(len(out_channel)):
            residual_channel = in_channel[i]
            for j in range(self.block_num):
                residual = ResBlock(residual_channel, out_channel[i])
                self.stage_res.append(residual)
        self.stage_res = nn.ModuleList(self.stage_res)

        for i in range(len(out_channel) if self.multi_scale_output else 1):
            for j in range(len(out_channel)):
                if j > i:
                    if not self.multi_scale_output:
                        fuse = BnActCon(
                            in_channel[j],
                            out_channel[j],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        )
                        self.fuse_res[str(fuse_index)] = fuse
                        fuse_index += 1
                    else:
                        fuse = BnActCon(
                            in_channel[j],
                            out_channel[i],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        )

                        self.fuse_res[str(fuse_index)] = fuse

                        fuse_index += 1
                elif j < i:
                    y = in_channel[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            fuse = BnActCon(
                                y, out_channel[i], kernel_size=1, stride=1, padding=0
                            )
                            self.fuse_res[str(fuse_index)] = fuse

                            fuse_index += 1
                        else:
                            fuse = BnActCon(
                                y, out_channel[j], kernel_size=1, stride=1, padding=0
                            )
                            self.fuse_res[str(fuse_index)] = fuse

                            fuse_index += 1
        self.fuse_res = nn.Sequential(self.fuse_res)

    def forward(self, x):
        stage_res = list()
        counter = 0

        for mod in range(len(x)):

            in_mod = x[mod]
            for res_block in range(self.block_num):
                in_mod = self.stage_res[res_block + counter](in_mod)
            stage_res.append(in_mod)
            counter += self.block_num
        x = stage_res

        fuse = list()
        fuse_index = 0

        for i in range(len(x) if self.multi_scale_output else 1):
            residual = x[i]
            for j in range(len(x)):
                if j > i:
                    if not self.multi_scale_output:
                        in_mod = self.fuse_res[fuse_index](x[j])
                        in_mod = F.interpolate(
                            in_mod, scale_factor=2 ** (j - i), mode="bilinear"
                        )
                        fuse.append(in_mod)
                        fuse_index += 1
                    else:
                        in_mod = self.fuse_res[fuse_index](x[j])
                        in_mod = F.interpolate(
                            in_mod, scale_factor=2 ** (j - i), mode="bilinear"
                        )
                        residual = residual + in_mod
                        fuse_index += 1

                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            y = self.fuse_res[fuse_index](y)
                            y = nn.MaxPool2d(kernel_size=2)(y)
                            fuse_index += 1
                        else:
                            y = self.fuse_res[fuse_index](y)
                            y = nn.MaxPool2d(kernel_size=2)(y)
                            fuse_index += 1

                    residual = residual + y
            fuse.append(residual)
        x = fuse
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.num_in = len(in_channel)
        self.num_out = len(out_channel)
        self.transition_layers = OrderedDict()
        transition_index = 0
        for i in range(self.num_out):
            if i < self.num_in:
                mod = BnActCon(
                    in_channel[i], out_channel[i], kernel_size=3, stride=1, padding=1
                )
                self.transition_layers[str(transition_index)] = mod
                transition_index += 1
            else:
                mod = BnActCon(
                    in_channel[-1], out_channel[i], kernel_size=3, stride=2, padding=1
                )
                self.transition_layers[str(transition_index)] = mod
                transition_index += 1

        self.transition_layers = nn.Sequential(self.transition_layers)

    def forward(self, x):
        transition_layers = list()
        transition_index = 0
        for i in range(self.num_out):
            if i < self.num_in:
                transition_layers.append(self.transition_layers[transition_index](x[i]))
                transition_index += 1
            else:
                transition_layers.append(
                    self.transition_layers[transition_index](x[-1])
                )
                transition_index += 1
        return transition_layers


class MapNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.channels_s2 = [64, 128]
        self.channels_s3 = [64, 128, 256]
        self.num_modules_s2 = 2
        self.num_modules_s3 = 3

        self.convolution_layer_1 = Convolution2d(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.activation = nn.ReLU(inplace=True)

        self.convolution_layer_2 = Convolution2d(64, 64)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.convolution_layer_3 = Convolution2d(64, 64)
        self.bn3 = BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.stage_1 = Step0()

        self.transition_layer_1 = TransitionLayer([256], self.channels_s2)
        self.stage_2 = Stage(self.channels_s2, self.channels_s2, self.num_modules_s2)

        self.transition_layer_2 = TransitionLayer([64, 128], self.channels_s3)

        self.stage_3 = Stage(self.channels_s3, self.channels_s3, self.num_modules_s3)
        self.stage_3_multi = Stage(
            self.channels_s3,
            self.channels_s3,
            self.num_modules_s3,
            multi_scale_output=False,
        )

        self.channel_squeeze = ChannelSELayer(448)
        self.spatial_pooling = SpatialPooling()
        self.new_feature = BnActCon(2368, 128, kernel_size=1, padding=0, stride=1)
        self.up1 = BnActCon(128, 64, kernel_size=3, padding=1, stride=1)
        self.up2 = BnActCon(64, 32, kernel_size=3, padding=1, stride=1)
        self.final = BnActCon(32, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x_1 = self.convolution_layer_1(x)
        x_1 = self.bn1(x_1)
        x_1 = self.activation(x_1)

        x_2 = self.convolution_layer_2(x_1)
        x_2 = self.bn2(x_2)
        x_2 = self.activation(x_2)

        x_3 = self.convolution_layer_3(x_2)
        x_3 = self.bn3(x_3)
        x_3 = self.activation(x_3)

        pool_1 = self.pool_1(x_3)
        stage_1 = self.stage_1(pool_1)

        trans_1 = self.transition_layer_1([stage_1])
        s2 = trans_1
        for i in range(self.num_modules_s2):
            s2 = self.stage_2(s2)

        trans_2 = self.transition_layer_2(s2)

        s3 = trans_2
        for i in range(self.num_modules_s3):
            if i == self.num_modules_s3 - 1:
                s3 = self.stage_3_multi(s3)

            else:
                s3 = self.stage_3(s3)

        cat_c3 = torch.cat(s3, 1)
        cat_c3_channel_squeeze = self.channel_squeeze(cat_c3)

        spatial = torch.cat([s3[0], s3[1]], 1)
        spatial = self.spatial_pooling(spatial)

        new_feature = torch.cat([cat_c3_channel_squeeze, spatial], 1)
        new_feature = self.new_feature(new_feature)

        up_1 = F.interpolate(new_feature, scale_factor=2, mode="bilinear")
        up_1 = self.up1(up_1)

        up_2 = F.interpolate(up_1, scale_factor=2, mode="bilinear")
        up_2 = self.up2(up_2)

        final = self.final(up_2)
        return final


# if __name__ == "__main__":
#     import torch
#
#     model = MapNet(
#         weight_path="/home/palnak/t.pt"
#     )
#     model.eval()
#     image = torch.randn(1, 3, 256, 256)
#     with torch.no_grad():
#         output = model.forward_propagate({"image": image})
#     if type(output) == list:
#         for i_1 in range(len(output)):
#             a = tuple(output[i_1].shape)
#             print(a)
#     else:
#         a = tuple(output.shape)
#         print(a)
