from torch import nn

from torchvision import models
from ml.ml_type.sem_seg.network_module import GCN, BoundaryRefine, UpSampleConvolution


class GCNModel(nn.Module):
    def __init__(
        self,
        backbone_to_use,
        pre_trained_image_net,
        top_layers_trainable=True,
        num_classes=1,
        deconvolution_method="bilinear",
    ):

        super().__init__()
        self.num_classes = num_classes
        self.deconvolution_method = deconvolution_method

        if hasattr(models, backbone_to_use):
            back_bone = getattr(models, backbone_to_use)(
                pretrained=pre_trained_image_net
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

        if not top_layers_trainable:
            for param in back_bone.parameters():
                param.requires_grad = False

        if backbone_to_use == "resnet50":
            layers_features = [256, 512, 1024, 2048]

        elif backbone_to_use == "resnet34":
            layers_features = [64, 128, 256, 512]

        else:
            raise NotImplementedError

        self.gcn_4 = GCN(in_planes=layers_features[-1], out_planes=self.num_classes)
        self.gcn_3 = GCN(in_planes=layers_features[-2], out_planes=self.num_classes)
        self.gcn_2 = GCN(in_planes=layers_features[-3], out_planes=self.num_classes)
        self.gcn_1 = GCN(in_planes=layers_features[-4], out_planes=self.num_classes)

        self.boundary_refine = BoundaryRefine(self.num_classes)

        self.up_sampling = UpSampleConvolution().init_method(
            method=self.deconvolution_method,
            down_factor=2,
            in_features=self.num_classes,
            num_classes=self.num_classes,
        )

    def forward(self, input_feature):
        if isinstance(input_feature, dict):
            x = input_feature["image"]
        else:
            raise TypeError

        res0 = self.layer0(x)
        res1 = self.layer1(res0)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        gcn1 = self.boundary_refine(self.gcn_1(res1))
        gcn2 = self.boundary_refine(self.gcn_2(res2))
        gcn3 = self.boundary_refine(self.gcn_3(res3))
        gcn4 = self.boundary_refine(self.gcn_4(res4))

        up_convolution_1 = self.up_sampling(gcn4)
        up_convolution_2 = self.up_sampling(
            self.boundary_refine(gcn3 + up_convolution_1)
        )
        up_convolution_3 = self.up_sampling(
            self.boundary_refine(gcn2 + up_convolution_2)
        )
        up_convolution_3 = self.up_sampling(
            self.boundary_refine(gcn1 + up_convolution_3)
        )

        final = self.boundary_refine(
            self.up_sampling(self.boundary_refine(up_convolution_3))
        )

        return final


# if __name__ == "__main__":
#     import torch
#
#     model = GCNModel(
#         backbone_to_use="resnet34",
#         pre_trained_image_net=True,
#         top_layers_trainable=True
#     )
#     model.eval()
#     image = torch.randn(2, 3, 512, 512)
#     image_temp = torch.randn(2, 3, 288, 288)
#     with torch.no_grad():
#         output = model.forward({"image": image})
#     a = tuple(output.shape)
#     print(a)
