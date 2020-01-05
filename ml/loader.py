import sys
import traceback

import torch
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader

from utils import print_format
import ml.ml_type as ml_type
from utils.directory_utils import directory_handler
from ml import scheduler as lr_helper


class ParameterLoader:
    def __init__(self, training_configuration, logger):
        self.training_configuration = training_configuration
        self.logger = logger
        self.model = self.load_model()

        self.train_data_set, self.val_data_set, self.test_data_set = (
            self.load_data_set()
        )
        self.train_loader, self.val_loader, self.test_loader = self.load_loader()
        self.validation = self.load_validation()

    def load_validation(self):
        try:
            problem_type = self.training_configuration["problem_type"]
            current_problem_type_loader = getattr(
                getattr(ml_type, problem_type), "Validation"
            )(problem_type)
            self.logger.log_info(
                "Validation - {} Problem Type loaded".format(problem_type)
            )

            return current_problem_type_loader
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_data_set(self):
        try:
            root = self.training_configuration["root"]
            model_input_dimension = self.training_configuration["initial_assignment"][
                "model_input_dimension"
            ]
            normalization = self.training_configuration["normalization"]
            transformation = self.training_configuration["transformation"]
            training_problem = self.training_configuration["problem_type"]

            train_data_set = getattr(getattr(ml_type, training_problem), "Dataloader")(
                root, model_input_dimension, "train", transformation, normalization
            )

            val_data_set = getattr(getattr(ml_type, training_problem), "Dataloader")(
                root, model_input_dimension, "val", transformation, normalization
            )

            test_data_set = getattr(getattr(ml_type, training_problem), "Dataloader")(
                root, model_input_dimension, "test", transformation, normalization
            )
            return train_data_set, val_data_set, test_data_set
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_loader(self):
        try:
            batch_size = self.training_configuration["initial_assignment"]["batch_size"]
            train_loader = DataLoader(
                dataset=self.train_data_set,
                shuffle=True,
                num_workers=0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )

            val_loader = DataLoader(
                dataset=self.val_data_set,
                shuffle=True,
                num_workers=0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )

            test_loader = DataLoader(
                dataset=self.test_data_set,
                shuffle=True,
                num_workers=0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )
            self.logger.log_info("Inference and Data loader - DataLoader complete")

            return train_loader, val_loader, test_loader
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    @staticmethod
    def get_gpu_device_ids():
        device_id = list()
        separator = ","
        gpu_device_available = torch.cuda.device_count()
        for i in range(gpu_device_available):
            device_id.append(str(i))
        device_id = separator.join(device_id)
        return device_id

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

    def load_parallel_model(self, model):
        try:
            if torch.cuda.is_available():
                device_ids = self.get_gpu_device_ids()
                if device_ids:
                    device_ids = list(map(int, device_ids.split(",")))
                else:
                    device_ids = None
                model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
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
            model_cuda = {
                [".".join(key.split(".")[1:])][0]: value
                for key, value in state["model"].items()
                if "module" in key.split(".")[0]
            }

            model.load_state_dict(model_cuda)
            model = self.load_parallel_model(model)
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

    def load_loss(self):
        try:
            loss_name = self.training_configuration["loss"]
            loss_param = self.training_configuration[loss_name]
            loss = getattr(ml_type, loss_name)(**loss_param)
            self.logger.log_info(
                "ParameterLoader - {} loss Loaded with parameters: {}".format(
                    loss_name, loss_param
                )
            )
            print_format.print_tab_fancy("Loss", "yellow")
            return loss
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_optimizer(self, model):
        try:
            optimizer_name = self.training_configuration["optimizer"]
            optimizer_param = self.training_configuration[optimizer_name]
            if optimizer_name == "SGD":
                optimizer = SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    **optimizer_param
                )
            elif optimizer_name == "Adam":
                optimizer = Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    **optimizer_param
                )
            elif optimizer_name == "RMSprop":
                optimizer = RMSprop(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    **optimizer_param
                )

            else:
                raise NotImplementedError
            self.logger.log_info(
                "ParameterLoader - {} optimizer Loaded with parameters: {}".format(
                    optimizer_name, optimizer_param
                )
            )
            print_format.print_tab_fancy("Optimizer", "yellow")
            return optimizer
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_lr_scheduler(self, optimizer):
        try:
            if bool(self.training_configuration["scheduler"]):
                lr_scheduler_name = self.training_configuration["scheduler"]
                lr_scheduler_param = self.training_configuration[lr_scheduler_name]

                lr_scheduler = getattr(lr_helper, lr_scheduler_name)(
                    **lr_scheduler_param, optimizer=optimizer
                )
                print_format.print_tab_fancy("Scheduler", "yellow")

            else:
                lr_scheduler = None
                lr_scheduler_name = None
                lr_scheduler_param = None
            self.logger.log_info(
                "ParameterLoader - {} Lr Scheduler Loaded with parameters: {}".format(
                    lr_scheduler_name, lr_scheduler_param
                )
            )

            return lr_scheduler
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))
