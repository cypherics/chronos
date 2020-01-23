from ml.base.base_network.base_pt_network import BaseNetwork
from ml.commons.network import DLinkNet34


class BinarySoftmaxDLinkNetExtractor(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = DLinkNet34(
            2,
            kwargs["res_net_to_use"],
            kwargs["pre_trained_image_net"],
        )

    def forward_propagate(self, input_feature):
        x = input_feature["image"]
        x_out = self.model(x)
        return x_out
