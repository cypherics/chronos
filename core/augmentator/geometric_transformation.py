import random
import cv2
import numpy as np

from utils.image_ops import get_random_crop_x_and_y, crop_image, perform_scale
from core.logger import debug, ChronosLogger

logger = ChronosLogger.get_logger()


class MirrorCrop:
    @debug
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            image_dim = img.shape
            dim = random.choice([384, 352, 416])
            height, width = get_random_crop_x_and_y((dim, dim), img.shape)
            img = crop_image(img, (dim, dim), (height, width))
            mask = crop_image(mask, (dim, dim), (height, width))
            limit = (image_dim[0] - dim) // 2
            img = cv2.copyMakeBorder(
                img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101
            )
            mask = cv2.copyMakeBorder(
                mask, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101
            )
        return img, mask


class RescaleCrop:
    @debug
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        dim_h, dim_w, _ = img.shape
        if random.random() < self.prob:
            # dim = random.choice([256, 300])
            dim = random.randint(384, 416)
            logger.debug("Running {} with dim {}".format(self.__class__.__name__, dim))
            height, width = get_random_crop_x_and_y((dim, dim), img.shape)
            img = crop_image(img, (dim, dim), (height, width))
            mask = crop_image(mask, (dim, dim), (height, width))
            img = perform_scale(img, (dim_h, dim_w), interpolation=cv2.INTER_NEAREST)
            mask = perform_scale(mask, (dim_h, dim_w), interpolation=cv2.INTER_NEAREST)
        return img, mask


class VerticalFlip:
    @debug
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    @debug
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    @debug
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            logger.debug("Running {} with flip {}".format(self.__class__.__name__, d))
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class RandomRotate90:
    @debug
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            logger.debug(
                "Running {} with factor {}".format(self.__class__.__name__, factor)
            )
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate:
    @debug
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                mat,
                (height, width),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            if mask is not None:
                mask = cv2.warpAffine(
                    mask,
                    mat,
                    (height, width),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

        return img, mask
