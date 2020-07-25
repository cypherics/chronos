import numpy as np
import cv2


def crop_image(img: np.ndarray, model_input_dimension: tuple, random_crop_coord: tuple):
    model_height, model_width = model_input_dimension
    height, width = random_crop_coord

    img = img[height : height + model_height, width : width + model_width]

    return img


def get_random_crop_x_and_y(model_input_dimension: tuple, image_input_dimension: tuple):
    model_height, model_width = model_input_dimension
    image_height, image_width, _ = image_input_dimension
    h_start = np.random.randint(0, image_height - model_height)
    w_start = np.random.randint(0, image_width - model_height)

    return h_start, w_start


def get_pad_limit(model_input_dimension: tuple, image_input_dimension: tuple):
    model_height, model_width = model_input_dimension
    image_height, image_width, _ = image_input_dimension

    limit = (model_height - image_height) // 2
    return limit


def pad_image(img: np.ndarray, limit: int):
    img = cv2.copyMakeBorder(
        img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101
    )
    return img


def perform_scale(img, dimension, interpolation=cv2.INTER_NEAREST):
    new_height, new_width = dimension
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


def handle_image_size(img, mask, dimension):
    if dimension < (img.shape[0], img.shape[1]):
        height, width = get_random_crop_x_and_y(dimension, img.shape)
        img = crop_image(img, dimension, (height, width))
        if mask is not None:
            mask = crop_image(mask, dimension, (height, width))
        return img, mask

    elif dimension > (img.shape[0], img.shape[1]):
        limit = get_pad_limit(dimension, img.shape)
        img = pad_image(img, limit)
        if mask is not None:
            mask = pad_image(mask, limit)
        return img, mask
    else:
        return img, mask


def load_image(path: str):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
