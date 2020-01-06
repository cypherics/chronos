from ml.ml_type.sem_seg.network_module import SubPixel
from ml.ml_type.sem_seg.network_module.bilinear_up_sampling import BiLinearUpSampling


class UpSampleConvolution:
    @staticmethod
    def init_method(method, down_factor, in_features, num_classes):
        if method == "bilinear":
            return BiLinearUpSampling(down_factor, in_features, num_classes)
        elif method == "sub_pixel":
            return SubPixel(down_factor, in_features, num_classes)
        else:
            raise NotImplementedError
