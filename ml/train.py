import os

from ml.commons.callbacks import (
    CallbackList,
    TensorBoardCallback,
    TrainStateCallback,
    TrainChkCallback,
    TimeCallback,
    TestCallback,
)
from ml.pt.factory import PtPlugin
from ml.pt.runner import PtRunner
from config.train_config import TrainConfig
from ml.pt.logger import info, DominusLogger
from utils.system_printer import SystemPrinter

CONFIG_RESTRICTION = ["DATASET", "MODEL", "TRAIN"]


class Train:
    def __init__(self, plugin, config_path):
        self.config_path = config_path
        self._train_data_loader, self._val_data_loader, self._test_data_loader = (
            None,
            None,
            None,
        )
        self._model = None
        self._criterion = None
        self._evaluator = None
        self._plugin = plugin

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

    def run(self):
        config = TrainConfig(self.config_path, CONFIG_RESTRICTION, self._plugin)
        save_path = os.path.join(config.training_path, config.version)

        config.write_config(save_path)
        DominusLogger().create_logger(
            config.root_folder,
            config.plugin,
            config.experiment_name,
            config.model,
            config.version,
        )
        plugin = PtPlugin(config)
        self.load_plugin(plugin, config)
        callbacks = self.register_callbacks(config)
        runner = PtRunner()
        runner.training(self, config, callbacks)

    @info
    def load_plugin(self, plugin, config):
        self.model = plugin.factory.create_network(config.model, config.model_param)
        SystemPrinter.sys_print(
            "\t LOADED MODEL - {}".format(self.model.__class__.__name__)
        )

        self.criterion = plugin.factory.create_criterion(config.loss, config.loss_param)
        SystemPrinter.sys_print(
            "\t LOADED CRITERION - {}".format(self.criterion.__class__.__name__)
        )

        self.evaluator = plugin.factory.create_evaluator()
        self.train_data_loader, self.val_data_loader, self.test_data_loader = (
            plugin.factory.create_data_set()
        )

    @info
    def register_callbacks(self, config):
        callbacks = CallbackList()
        callbacks.append(
            TensorBoardCallback(os.path.join(config.training_path, config.version))
        )
        callbacks.append(TrainStateCallback(config.default_state, config.best_state))
        callbacks.append(TrainChkCallback(config.chk_pth))
        callbacks.append(TimeCallback())
        if self.evaluator is not None:
            callbacks.append(
                TestCallback(
                    test_loader=self.test_data_loader,
                    evaluator=self.evaluator,
                    pth=os.path.join(config.training_path, config.version),
                )
            )
        return callbacks
