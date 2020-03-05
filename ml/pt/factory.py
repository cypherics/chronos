import os

from ml.ml_type.base.base_factory import Factory
from utils.logger import LogDecorator


class Plugin:
    def __init__(self, config):
        self.config = config
        self.factory = self.create_factory()

        self._train_data_loader, self._val_data_loader, self._test_data_loader = (
            None,
            None,
            None,
        )
        self._model = None
        self._criterion = None
        self._evaluator = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, value):
        self._criterion = value

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, value):
        self._evaluator = value

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @train_data_loader.setter
    def train_data_loader(self, value):
        self._train_data_loader = value

    @property
    def val_data_loader(self):
        return self._val_data_loader

    @val_data_loader.setter
    def val_data_loader(self, value):
        self._val_data_loader = value

    @property
    def test_data_loader(self):
        return self._test_data_loader

    @test_data_loader.setter
    def test_data_loader(self, value):
        self._test_data_loader = value

    @LogDecorator()
    def create_factory(self):
        """
        Load module for dynamic configuration of plugin factory.
        It loads the plugin python file by looking into dir named as 'config['framework']['plugin'] and
        factory file named as '{config['framework']['plugin']}_factory.py'
        """
        plugin_name = (
            self.config.problem_type
        )  # TODO self.configuration["problem_type"]
        plugin = os.path.join("ml/ml_type/", plugin_name)

        file = os.path.join(os.curdir, plugin, plugin_name + "_factory.py")

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            self.config.problem_type, file
        )  # TODO self.configuration["problem_type"]
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        for cls in Factory.__subclasses__():
            f = cls(self.config)
            return f

    @LogDecorator()
    def load_trainer_plugin(self):
        self.model = self.factory.create_network(
            self.config.model, self.config.model_param
        )
        self.criterion = self.factory.create_criterion(
            self.config.loss, self.config.loss_param
        )

        self.evaluator = self.factory.create_evaluator()
        self.train_data_loader, self.val_data_loader, self.test_data_loader = (
            self.factory.create_data_set()
        )
