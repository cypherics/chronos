import os
from ml.commons.utils.model_util import (
    get_current_state,
    set_model_state,
    set_optimizer_state,
    load_parallel_model,
)
from ml.pt.logger import debug, DominusLogger
from utils.system_printer import SystemPrinter

logger = DominusLogger.get_logger()


class PtState:
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
    def state_obj_epoch(self):
        return {"my_state": self.compress_state_obj("complete")}

    @property
    def state_obj_interruption(self):
        return {"my_state": self.compress_state_obj("interrupt")}

    def compress_state_obj(self, run_state):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if self.optimizer is not None
            else "NA",
            "starting_epoch": self.starting_epoch + 1
            if run_state == "complete"
            else self.starting_epoch,
            "step": self.step,
            "bst_vld_loss": self.bst_vld_loss
            if self.bst_vld_loss is not None
            else "NA",
        }

    @debug
    def extract_state(self, pth):
        if os.path.exists(pth):
            logger.debug("Loading Existing State {}".format(pth))
            ongoing_state = get_current_state(pth)
            if self.check_key_and_none(ongoing_state, "model"):
                self.model = set_model_state(self.model, ongoing_state["model"])
                logger.debug("Existing Model Loaded")

            self.model = load_parallel_model(self.model)

            if self.check_key_and_none(ongoing_state, "optimizer"):
                self.optimizer = set_optimizer_state(
                    self.optimizer, ongoing_state["optimizer"]
                )
                logger.debug(
                    "Existing Optimizer Loaded with lr {}".format(
                        self.optimizer.param_groups[0]["lr"]
                    )
                )

            if self.check_key_and_none(ongoing_state, "starting_epoch"):
                self.starting_epoch = ongoing_state["starting_epoch"]
                logger.debug("Existing Start Epoch {}".format(self.starting_epoch))

            if self.check_key_and_none(ongoing_state, "step"):
                self.step = ongoing_state["step"]
                logger.debug("Existing Step Epoch {}".format(self._step))

            if self.check_key_and_none(ongoing_state, "bst_vld_loss"):
                self.bst_vld_loss = ongoing_state["bst_vld_loss"]
                logger.debug("Existing Best Valid Loss {}".format(self.bst_vld_loss))

        else:
            self.model = load_parallel_model(self.model)

    @staticmethod
    def check_key_and_none(state, key):
        if key not in state:
            SystemPrinter.sys_print("{} not Found, Setting to default".format(key))
            return False
        elif state[key] is None:
            SystemPrinter.sys_print("None {} Found, Setting to default".format(key))
            return False
        else:
            return True
