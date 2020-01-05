from collections import namedtuple

import torch
from torchvision.models import vgg16_bn


class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_model = vgg16_bn(pretrained=True)
        self.vgg_model.cuda() if torch.cuda.is_available() else self.vgg_model
        self.vgg_model.eval()
        self.vgg_layers = self.vgg_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(0, 6):
            self.slice1.add_module(str(x), self.vgg_layers[x])
        for x in range(6, 13):
            self.slice2.add_module(str(x), self.vgg_layers[x])
        for x in range(13, 23):
            self.slice3.add_module(str(x), self.vgg_layers[x])
        for param in self.vgg_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h

        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)
        return out


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class PerceptualLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(PerceptualLoss, self).__init__()
        if weight is None:
            weight = [0.2, 0.7, 0.1]
        self.vgg_model = VGG()
        self.weight = weight
        self.l1_loss = torch.nn.L1Loss()

    @staticmethod
    def normalize_batch(batch, divide_by_255=True):
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if divide_by_255:
            batch = batch.div_(255.0)
        return (batch - mean) / std

    def forward(self, input, target, sum_layers=True):

        input_tensor = input.clone()
        target_tensor = target.clone()
        res = self.l1_loss(input_tensor, target_tensor) / 100

        input = self.normalize_batch(input)
        target = self.normalize_batch(target, divide_by_255=False)

        input_features = self.vgg_model(input)
        target_features = self.vgg_model(target)

        res += (
            self.l1_loss(input_features.relu1_2, target_features.relu1_2)
            * self.weight[0]
        )
        res += (
            self.l1_loss(input_features.relu2_2, target_features.relu2_2)
            * self.weight[1]
        )

        res += (
            self.l1_loss(input_features.relu3_3, target_features.relu3_3)
            * self.weight[2]
        )
        return res


class ResolutionPerceptualLoss:
    def __init__(self, weight):
        self.weight = weight
        self.perceptual_loss = PerceptualLoss(weight=self.weight)

    def __call__(self, outputs, **kwargs):
        target = kwargs["label"]
        return self.perceptual_loss(outputs, target)
