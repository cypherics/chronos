from ml.ml_module import SubPixel
from ml.ml_module import BiLinearUpSampling


class UpSampleConvolution:
    @staticmethod
    def init_method(method, down_factor, in_features, num_classes):
        if method == "bilinear":
            return BiLinearUpSampling(down_factor, in_features, num_classes)
        elif method == "sub_pixel":
            return SubPixel(down_factor, in_features, num_classes)
        else:
            raise NotImplementedError
