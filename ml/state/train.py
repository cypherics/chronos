import os
from ml.commons.utils.model_utility import (
    get_current_state,
    set_model_state,
    set_optimizer_state,
    load_parallel_model,
)


class TrainState(object):
    def __init__(self):
        self._model = None
        self._optimizer = None
        self._starting_epoch = 1
        self._step = 1
        self._bst_vld_loss = None
        self._state_obj = dict()

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
    def starting_epoch(self):
        return self._starting_epoch

    @starting_epoch.setter
    def starting_epoch(self, value):
        self._starting_epoch = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    @property
    def bst_vld_loss(self):
        return self._step

    @bst_vld_loss.setter
    def bst_vld_loss(self, value):
        self._step = value

    @property
    def state_obj(self):
        return {"my_state": self.compress_state_obj()}

    def compress_state_obj(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "starting_epoch": self.starting_epoch,
            "step": self.step,
            "bst_vld_loss": self.bst_vld_loss,
        }

    def extract_state(self, pth):
        if os.path.exists(pth):
            ongoing_state = get_current_state(pth)
            self.model = set_model_state(self.model, ongoing_state["model"])
            self.model = load_parallel_model(self.model)

            self.optimizer = set_optimizer_state(
                self.optimizer, ongoing_state["optimizer"]
            )
            self.starting_epoch = ongoing_state["starting_epoch"]
            self.step = ongoing_state["step"]
            self.bst_vld_loss = ongoing_state["bst_vld_loss"]
        else:
            self.model = load_parallel_model(self.model)
