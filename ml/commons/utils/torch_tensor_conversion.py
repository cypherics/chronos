import torch

import numpy as np


def cuda_variable(x):
    """

    :param x:
    :return:
    """
    if isinstance(x, (list, tuple)):
        return [cuda_variable(y) for y in x]

    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = cuda_variable(v)
        return x

    return cuda(x)


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


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
    data = cuda_variable(data)
    return data
