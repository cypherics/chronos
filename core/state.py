import torch

from utils.network_util import load_parallel_model, adjust_model
from core.logger import ChronosLogger, info, debug
from utils.system_printer import SystemPrinter

logger = ChronosLogger.get_logger()


class PtState:
    def __init__(self):
        self._model = None
        self._optimizer = None
        self._starting_epoch = None
        self._step = None
        self._bst_vld_loss = None

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
        return self._bst_vld_loss

    @bst_vld_loss.setter
    def bst_vld_loss(self, value):
        self._bst_vld_loss = value

    @property
    def epoch_state(self):
        return {"my_state": self.collect_state("complete")}

    @property
    def interruption_state(self):
        return {"my_state": self.collect_state("interrupt")}

    def collect_state(self, run_state):
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

    @info
    def restart(self, model, optimizer, state_pth):
        SystemPrinter.sys_print("\t Loading Existing State {}".format(state_pth))
        ongoing_state = self.extract_state(state_pth)
        if self.check_key_and_none(ongoing_state, "model"):
            self.model = self.set_model_state(model, ongoing_state["model"])
            logger.debug("Existing Model Loaded")

        self.model = load_parallel_model(self.model)

        if self.check_key_and_none(ongoing_state, "optimizer"):
            self.optimizer = self.set_optimizer_state(
                optimizer, ongoing_state["optimizer"]
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

    @info
    def new(self, model, optimizer):
        SystemPrinter.sys_print("\t Loading New State")
        self.model = model
        self.optimizer = optimizer
        self.model = load_parallel_model(self.model)
        self.starting_epoch = 1
        self.step = 1
        self.bst_vld_loss = None

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

    @staticmethod
    @debug
    def extract_state(weight_path):
        state = torch.load(str(weight_path), map_location="cpu")
        return state

    @staticmethod
    def set_model_state(model, model_state):
        if model_state is not None:
            model_adjusted = adjust_model(model_state)
            model.load_state_dict(model_adjusted)
        return model

    @staticmethod
    def set_optimizer_state(optimizer, optimizer_state):
        optimizer.load_state_dict(optimizer_state)
        return optimizer
