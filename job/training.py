import os

from ml.pt.factory import Plugin
from ml.trainer import Trainer
from config.train_config import TrainConfig
from ml.pt.logger import create_logger, PtLogger

CONFIG_RESTRICTION = ["DATASET", "MODEL", "TRAIN"]


class Training:
    def __init__(self, config_path):
        self.config_path = config_path
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

    def default(self):
        config = TrainConfig.create_config(self.config_path, CONFIG_RESTRICTION)
        config.set_property("training_state", "DEFAULT")
        config.set_additional_property()

        save_path = os.path.join(config.training_path, config.version)

        config.write_config(save_path)
        create_logger(config.root_folder, config.experiment_name)
        plugin = Plugin(config)
        self.load_trainer_plugin(plugin, config)
        trainer = Trainer()
        trainer.training(self, config)

    def resume(self):
        config = TrainConfig.create_config(self.config_path, CONFIG_RESTRICTION)
        config.set_run_property(os.path.dirname(self.config_path))
        config.update_property("training_state", "RESUME")

        save_path = os.path.join(config.training_path, config.version)

        config.write_config(save_path)
        create_logger(config.root_folder, config.experiment_name)
        plugin = Plugin(config)
        self.load_trainer_plugin(plugin, config)
        trainer = Trainer()
        trainer.training(self, config)

    @PtLogger()
    def load_trainer_plugin(self, plugin, config):
        self.model = plugin.factory.create_network(config.model, config.model_param)
        self.criterion = plugin.factory.create_criterion(config.loss, config.loss_param)

        self.evaluator = plugin.factory.create_evaluator()
        self.train_data_loader, self.val_data_loader, self.test_data_loader = (
            plugin.factory.create_data_set()
        )
