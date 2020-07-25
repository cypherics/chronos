from ml.network import MapNet
from plugins.base.network.base_network import BaseNetwork


class BinaryNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "weight_path" in kwargs.keys():
            self.transfer = True
            self.weight_path = kwargs["weight_path"]
        else:
            self.transfer = False
        self.map_net = MapNet()
        if self.transfer:
            self.load_pre_trained(self.weight_path)

    def forward_propagate(self, x) -> dict:
        x = x["image"]
        return {"output": self.map_net(x)}
