import torch

import numpy as np


def make_cuda(x):
    """

    :param x:
    :return:
    """
    if isinstance(x, (list, tuple)):
        return [make_cuda(y) for y in x]

    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = make_cuda(v)
        return x

    return x.cuda() if torch.cuda.is_available() else x


def to_input_image_tensor(img):
    if isinstance(img, list):
        return [to_input_image_tensor(image) for image in img]
    return to_tensor(np.moveaxis(img, -1, 0))


def to_label_image_tensor(mask):
    return to_tensor(np.expand_dims(mask, 0))


def to_tensor(data):
    return torch.from_numpy(data).float()


def to_multi_output_label_image_tensor(mask):
    return to_tensor(np.moveaxis(mask, -1, 0))


def add_extra_dimension(data):
    if isinstance(data, (list, tuple)):
        return [torch.unsqueeze(y.cuda(), dim=0) for y in data]
    return torch.unsqueeze(data.cuda(), dim=0)


def prediction_tensor_cuda(data):
    data = to_input_image_tensor(data)
    data = add_extra_dimension(data)
    data = make_cuda(data)
    return data


def convert_tensor_to_numpy(ip):
    if ip.is_cuda:
        return ip.data.cpu().numpy()
    else:
        return ip.data.numpy()
