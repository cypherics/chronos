from plugins.base.base_factory import Factory
from plugins.binary import network, criterion
from plugins.binary.binary_data_set import BinaryDataSet
from plugins.binary.binary_evaluator import BinaryEvaluator
from plugins.binary.binary_callbacks import BinaryCallback


class BinaryFactory(Factory):
    def __init__(self, config):
        self.config = config
        super(BinaryFactory, self).__init__(config)

    def create_data_set(self):
        return BinaryDataSet.get_data(self.config)

    def create_evaluator(self):
        return BinaryEvaluator()

    def create_criterion(self, criterion_name, criterion_param):
        criterion_fn = getattr(criterion, criterion_name)(**criterion_param)
        return criterion_fn

    def create_network(self, model_name, model_param):
        model = getattr(network, model_name)(**model_param)
        return model

    def create_callback(self):
        return BinaryCallback(self.config).get_callbacks()
