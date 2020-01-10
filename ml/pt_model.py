import sys
import traceback

import torch


from utils.directory_utils import directory_handler
import ml.ml_type as ml_type
from utils.multi_gpu import adjust_model_keys, get_gpu_device_ids


class ModelPt:
    def __init__(self, training_configuration, logger):
        self.training_configuration = training_configuration
        self.logger = logger
        self.model = self.load_model()

    def load_model(self):
        try:
            model_name = self.training_configuration["model"]
            model_param = self.training_configuration[
                self.training_configuration["model"]
            ]
            model = getattr(ml_type, model_name)(**model_param)
            self.logger.log_info(
                "Model Loader -  {} load success with parameters: {}".format(
                    model_name, model_param
                )
            )

            return model
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_model_weights_and_params(
        self, optimizer, status="DEFAULT", weight_path=None
    ):
        if status == "TRANSFER_LEARNING":
            transfer_weights_path = self.training_configuration["transfer_weights_path"]
            state = self.get_current_state(transfer_weights_path)
            model = self.load_current_model_state(self.model, state)
            starting_epoch = 0
            step = 0
            learning_rate = optimizer.defaults["lr"]
            optimizer = optimizer

        elif status == "RESUME":
            resume_weight_path = weight_path
            state = self.get_current_state(resume_weight_path)
            model = self.load_current_model_state(self.model, state)
            starting_epoch = state["starting_epoch"] if "starting_epoch" in state else 0
            step = state["step"] if "step" in state else 0
            optimizer.load_state_dict(state["optimizer"])
            learning_rate = optimizer.defaults["lr"]

        elif status == "DEFAULT":
            model = self.model
            model = self.load_parallel_model(model)
            optimizer = optimizer
            starting_epoch = 0
            step = 0
            learning_rate = optimizer.defaults["lr"]

        else:
            raise NotImplementedError
        self.logger.log_info(
            "Model Loader - Weights loaded with starting_epoch : {}".format(
                starting_epoch
            )
        )
        return model, optimizer, learning_rate, starting_epoch, step

    def get_model_version_and_status(self, training_path):
        status = (
            "TRANSFER_LEARNING"
            if bool(self.training_configuration["transfer_weights_path"])
            else self.training_configuration["training_state"]
        )
        current_version, previous_version = directory_handler.make_version(
            training_path
        )
        if status == "RESUME":
            current_version = previous_version
        self.logger.log_info(
            "Model Loader -  status : {}, version: {}".format(status, current_version)
        )
        try:
            assert status in ["TRANSFER_LEARNING", "RESUME", "DEFAULT"]
        except AssertionError as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

        return status, current_version

    @staticmethod
    def save_check_point(model, optimizer, epoch, step, lr, model_path):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "starting_epoch": epoch,
                "step": step,
                "lr": lr,
            },
            str(model_path),
        )

    def load_parallel_model(self, model):
        try:
            if torch.cuda.is_available():
                device_ids = get_gpu_device_ids()
                if device_ids:
                    device_ids = list(map(int, device_ids.split(",")))
                else:
                    device_ids = None
                model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
                return model
            else:
                return model
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def get_current_state(self, weight_path):
        try:
            state = torch.load(str(weight_path))
            return state
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_current_model_state(self, model, state):
        try:
            model_cuda = adjust_model_keys(state)
            model.load_state_dict(model_cuda)
            model = self.load_parallel_model(model)
            return model
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))
