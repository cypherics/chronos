import numpy as np

from sklearn.preprocessing import MinMaxScaler

from core.logger import debug


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray, range_between_0_1=False) -> np.ndarray:
        if not range_between_0_1:
            max_pixel_value = 255.0
            img = img.astype(np.float32) / max_pixel_value

        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std
        return img


class MinMax:
    @debug
    def __init__(self):
        self.normalization = "minmax"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = np.zeros_like(img).astype(np.float32)

        for i in range(img.shape[2]):
            c = img[:, :, i].min()
            d = img[:, :, i].max()

            t = (img[:, :, i] - c) / (d - c)
            out[:, :, i] = t
        out.astype(np.float32)

        return out.astype(np.float32)


class MinMaxImageNet:
    @debug
    def __init__(self):
        self.normalization = "minmax"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = np.zeros_like(img).astype(np.float32)
        for i in range(img.shape[2]):
            c = img[:, :, i].min()
            d = img[:, :, i].max()

            t = (img[:, :, i] - c) / (d - c)
            out[:, :, i] = t
        out.astype(np.float32)

        out = Normalize().__call__(out, True)
        return out.astype(np.float32)


class MinMaxScaleImageNet:
    @debug
    def __init__(self):
        self.normalization = "minmax"
        self.minmax_scaler = MinMaxScaler()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w, b = img.shape
        img = img.reshape(h * w, b)
        img = img.astype(np.float64)
        img = self.minmax_scaler.fit_transform(img)
        img = img.reshape(h, w, b)
        img = Normalize().__call__(img, True)
        return img.astype(np.float32)


class DivideBy255:
    @debug
    def __init__(self):
        self.normalization = "divide_by_255"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255
        return img


class InriaData:
    @debug
    def __init__(self):
        self.mean = (0.42068335885143315, 0.43821200008781647, 0.4023395608370018)
        self.std_dev = (0.03871459540580076, 0.039615887087616986, 0.04203108867447648)

        self.normalize = Normalize(mean=self.mean, std=self.std_dev)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255
        img = self.normalize.__call__(img, True)
        return img
