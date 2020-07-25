import random
import cv2
import numpy as np

from core.logger import debug, ChronosLogger

logger = ChronosLogger.get_logger()


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class CLAHE:
    @debug
    def __init__(self, prob=0.5, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.prob = prob

    def __call__(self, im, mask):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            img_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            clahe_method = cv2.createCLAHE(
                clipLimit=self.clipLimit, tileGridSize=self.tileGridSize
            )
            img_yuv[:, :, 0] = clahe_method.apply(img_yuv[:, :, 0])
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            im = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
        return im, mask


class RandomHueSaturationValue:
    @debug
    def __init__(
        self,
        hue_shift_limit=(-10, 10),
        sat_shift_limit=(-25, 25),
        val_shift_limit=(-25, 25),
        prob=0.5,
    ):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(
                self.hue_shift_limit[0], self.hue_shift_limit[1]
            )
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(
                self.sat_shift_limit[0], self.sat_shift_limit[1]
            )
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(
                self.val_shift_limit[0], self.val_shift_limit[1]
            )
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, mask


class RandomContrast:
    @debug
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            alpha = 1.0 + self.limit * random.uniform(-1, 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, mask


class RandomBrightness:
    @debug
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img, mask


class RandomFilter:
    """
    blur sharpen, etc
    """

    @debug
    def __init__(self, limit=0.5, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            logger.debug("Running {}".format(self.__class__.__name__))
            alpha = self.limit * random.uniform(0, 1)
            kernel = np.ones((3, 3), np.float32) / 9 * 0.2

            colored = img[..., :3]
            colored = alpha * cv2.filter2D(colored, -1, kernel) + (1 - alpha) * colored
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(colored, dtype, maxval)

        return img, mask
