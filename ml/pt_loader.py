import sys
import traceback

import torch
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader

from ml.base import BaseDataLoader, BaseValidation
from utils import print_format
import ml.ml_type as ml_type
from ml import scheduler as lr_helper


class LoaderPt:
    def __init__(self, training_configuration, logger):
        self.training_configuration = training_configuration
        self.logger = logger

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
            )()
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
