import os

import torch

from core.extensions.callbacks import (
    CallbackList,
    TensorBoardCallback,
    TrainStateCallback,
    TrainChkCallback,
    TimeCallback,
)

from core.factory import Plugin
from core.learner import Learner
from config import Config
from core.logger import info, ChronosLogger
from core.extensions.metric import MetricList

CONFIG_RESTRICTION = ["DATASET", "MODEL", "TRAIN"]


class Train:
    def __init__(self, plugin, config_path):
        self.config_path = config_path
        self._optimizer = None
        self._plugin = None
        self._plugin_name = plugin

    @property
    def plugin(self):
        return self._plugin

    @plugin.setter
    def plugin(self, value):
        self._plugin = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def run(self):
        config = Config(self.config_path, CONFIG_RESTRICTION, self._plugin_name)
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
        self.plugin = Plugin(config)
        self.plugin.load_plugin()
        self.load_optimizer(config.optimizer_name, config.optimizer_param)

        callbacks = self.register_callbacks(config, self.plugin.extension.callbacks())
        metrics = self.register_metrics(self.plugin.extension.metrics())
        Learner(config).training(self.plugin, self.optimizer, callbacks, metrics)

    @info
    def load_optimizer(self, optimizer_name, optimizer_param):
        self.optimizer = getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, self.plugin.model.parameters()),
            **optimizer_param
        )

    @info
    def register_callbacks(self, config, extension_callbacks):
        callbacks = CallbackList()
        callbacks.append(
            TensorBoardCallback(os.path.join(config.training_path, config.version))
        )
        callbacks.append(TrainStateCallback(config.default_state, config.best_state))
        callbacks.append(TrainChkCallback(config.chk_pth))
        callbacks.append(TimeCallback())

        for individual_callbacks in extension_callbacks:
            callbacks.append(individual_callbacks)

        return callbacks

    @info
    def register_metrics(self, metric):
        metrics = MetricList(metric)
        return metrics
