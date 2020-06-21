from torch import nn
from ml.modules import UpSampleConvolution

__reference__ = "https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/refine_net.py"
__paper__ = "https://arxiv.org/pdf/1611.06612.pdf"


"""
backbone_features - represent the output of backbone network 
refine_block_features - represent the output of Refine Block

"""


def convolution_1x1(in_planes, out_planes, stride=1, padding=0, bias=False):
    """1x1 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )


class MultiResolutionFusion(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.convolution_layer_lower_inputs = convolution_1x1(in_planes, out_planes)
        self.convolution_layer_higher_inputs = convolution_1x1(out_planes, out_planes)
        self.up_sample = UpSampleConvolution().init_method(
            method="bilinear",
            down_factor=2,
            in_features=out_planes,
            num_classes=out_planes,
        )

    def forward(self, backbone_features, refine_block_features=None):
        if refine_block_features is None:
            """
            refine_block_features is None :
                Suggests RefineNet-4
            """
            return self.convolution_layer_higher_inputs(backbone_features)
        else:
            backbone_features = self.convolution_layer_higher_inputs(backbone_features)

            refine_block_features = self.convolution_layer_lower_inputs(
                refine_block_features
            )
            refine_block_features = self.up_sample(refine_block_features)

            return refine_block_features + backbone_features


class ChainedResidualPooling(nn.Module):
    """
    Section-1:
        Chained residual pooling is able to capture background context from a large image region
    """

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.convolution_layer_1 = convolution_1x1(
            in_planes=in_planes, out_planes=out_planes
        )
        self.max_pooling_layer = nn.MaxPool2d((5, 5), stride=1, padding=2)

    def forward(self, x):
        x_non_linearity = self.non_linearity(x)
        first_pass = self.max_pooling_layer(x_non_linearity)
        first_pass = self.convolution_layer_1(first_pass)

        intermediate_sum = first_pass + x_non_linearity

        second_pass = self.max_pooling_layer(first_pass)
        second_pass = self.convolution_layer_1(second_pass)

        x = second_pass + intermediate_sum
        return x


class RefineBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.multi_resolution_fusion = MultiResolutionFusion(in_planes, out_planes)
        self.chained_residual_pooling = ChainedResidualPooling(out_planes, out_planes)

    def forward(self, backbone_features, refine_block_features=None):
        """

        :param backbone_features: input from backbone network
        :param refine_block_features: input from refine net block
        :return:
        """
        if refine_block_features is None:
            """
            refine_block_features is None :
                Suggests RefineNet-4
            """
            """
            section 3.2 - 
                Multi-resolution fusion :   
                    If there is only one input path (e.g , the case of RefineNet-4 the input will directly go thorough
            """
            x = self.multi_resolution_fusion(backbone_features)
            x = self.chained_residual_pooling(x)

            return x
        else:

            x = self.multi_resolution_fusion(backbone_features, refine_block_features)
            x = self.chained_residual_pooling(x)

            return x


class ReFineNetLite(nn.Module):
    def __init__(self, backbone_to_use_features):
        super().__init__()

        """
        section 3.1 -
            In practice each ResNet output is passed through one convolution layer to adapt the dimensionality
        """
        self.convolution_layer_4_dim_reduction = convolution_1x1(
            in_planes=backbone_to_use_features[-1], out_planes=512
        )
        self.convolution_layer_3_dim_reduction = convolution_1x1(
            in_planes=backbone_to_use_features[-2], out_planes=256
        )
        self.convolution_layer_2_dim_reduction = convolution_1x1(
            in_planes=backbone_to_use_features[-3], out_planes=256
        )
        self.convolution_layer_1_dim_reduction = convolution_1x1(
            in_planes=backbone_to_use_features[-4], out_planes=256
        )

        self.refine_block_4 = RefineBlock(in_planes=512, out_planes=512)
        self.refine_block_3 = RefineBlock(in_planes=512, out_planes=256)
        self.refine_block_2 = RefineBlock(in_planes=256, out_planes=256)
        self.refine_block_1 = RefineBlock(in_planes=256, out_planes=256)

    def forward(self, encoder_output: list):

        layer_1_output = encoder_output[0]  # 1/4
        layer_2_output = encoder_output[1]  # 1/8
        layer_3_output = encoder_output[2]  # 1/16
        layer_4_output = encoder_output[3]  # 1/32

        backbone_layer_4 = self.convolution_layer_4_dim_reduction(
            layer_4_output
        )  # 1/32
        backbone_layer_3 = self.convolution_layer_3_dim_reduction(
            layer_3_output
        )  # 1/16
        backbone_layer_2 = self.convolution_layer_2_dim_reduction(layer_2_output)  # 1/8
        backbone_layer_1 = self.convolution_layer_1_dim_reduction(layer_1_output)  # 1/4

        refine_block_4 = self.refine_block_4(backbone_layer_4)
        refine_block_3 = self.refine_block_3(backbone_layer_3, refine_block_4)
        refine_block_3 = self.refine_block_2(backbone_layer_2, refine_block_3)
        refine_block_1 = self.refine_block_1(backbone_layer_1, refine_block_3)

        return refine_block_1
