import os

import torch

from core.callbacks import (
    CallbackList,
    TensorBoardCallback,
    TrainStateCallback,
    TrainChkCallback,
    TimeCallback,
    TestCallback,
)

from core.factory import PtPlugin
from core.runner import PtRunner
from config import Config
from core.logger import info, ChronosLogger

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
        self._optimizer = None
        self._plugin = plugin

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

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
        config = Config(self.config_path, CONFIG_RESTRICTION, self._plugin)
        if not config.resume:
            config.generate_additional_train_property()
            save_path = os.path.join(config.training_path, config.version)
            config.write_config(save_path)

        ChronosLogger().create_logger(
            config.root_folder,
            config.plugin,
            config.experiment_name,
            config.model_name,
            config.version,
        )
        plugin = PtPlugin(config)
        (
            self.model,
            self.criterion,
            self.evaluator,
            self.train_data_loader,
            self.val_data_loader,
            self.test_data_loader,
        ) = plugin.load_plugin(config)
        self.load_optimizer(config.optimizer_name, config.optimizer_param)

        callbacks = self.register_callbacks(config)
        runner = PtRunner(config)
        runner.training(
            self.model,
            self.optimizer,
            self.criterion,
            self.train_data_loader,
            self.val_data_loader,
            callbacks,
        )

    @info
    def load_optimizer(self, optimizer_name, optimizer_param):
        self.optimizer = getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **optimizer_param
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
