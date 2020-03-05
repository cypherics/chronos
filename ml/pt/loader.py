import os
import sys

import torch

from ml.commons.utils.multi_gpu import (
    load_parallel_model,
    set_model_state,
    get_current_state,
)
from utils.logger import LogDecorator
from ml.commons import scheduler as lr_helper


class Loader:
    def __init__(self, config):
        self.config = config

    @LogDecorator()
    def default(self, model, optimizer):
        model = load_parallel_model(model)
        optimizer = optimizer
        starting_epoch = 0
        step = 0
        learning_rate = optimizer.defaults["lr"]
        sys.stdout.write(
            "DEFAULT State Loaded at epoch {}, Lr {}".format(starting_epoch, step)
        )
        return model, optimizer, starting_epoch, step, learning_rate

    @LogDecorator()
    def resume(self, model, optimizer):
        saved_trainer_state = self.get_existing_state(self.config.chk_path)
        model = set_model_state(model, saved_trainer_state)
        model = load_parallel_model(model)
        starting_epoch = saved_trainer_state["starting_epoch"]
        step = saved_trainer_state["step"]
        optimizer.load_state_dict(saved_trainer_state["optimizer"])
        learning_rate = optimizer.defaults["lr"]
        sys.stdout.write(
            "RESUME State Loaded at epoch {}, Lr {}".format(starting_epoch, step)
        )
        return model, optimizer, starting_epoch, step, learning_rate

    @LogDecorator()
    def get_existing_state(self, weight_path):
        assert os.path.exists(weight_path), "NO STATE FOUND TO LOAD"
        return get_current_state(weight_path)

    @LogDecorator()
    def load_state(self, model, optimizer, state):
        func = getattr(self, state.lower())
        return func(model, optimizer)

    @LogDecorator()
    def load_optimizer(self, model):
        optimizer_name = self.config.optimizer
        optimizer_param = self.config.optimizer_param
        return getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, model.parameters()), **optimizer_param
        )

    @LogDecorator()
    def load_lr_scheduler(self, optimizer):
        if bool(self.config.scheduler):
            lr_scheduler_name = self.config.scheduler
            lr_scheduler_param = self.config.scheduler_param

            lr_scheduler = getattr(lr_helper, lr_scheduler_name)(
                **lr_scheduler_param, optimizer=optimizer
            )
        else:
            lr_scheduler = None
            lr_scheduler_name = None
            lr_scheduler_param = None

        return lr_scheduler
