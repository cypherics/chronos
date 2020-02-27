import torch

from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader

from ml.commons.utils.multi_gpu import get_gpu_device_ids, adjust_model_keys, get_current_state, load_parallel_model
from utils.logger import LogDecorator
from ml.commons import scheduler as lr_helper


class Loader:
    def __init__(self, plugin, training_configuration):
        self.plugin = plugin
        self.training_configuration = training_configuration

    @LogDecorator()
    def load_training_state(
        self, model, optimizer, status="DEFAULT", weight_path=None
    ):
        if status == "INFERENCE":
            state = get_current_state(weight_path)
            model = self.load_current_model_state(model, state)
            starting_epoch = 0
            step = 0
            learning_rate = None
            optimizer = None

        elif status == "TRANSFER_LEARNING":
            transfer_weights_path = self.training_configuration["transfer_weights_path"]
            state = get_current_state(transfer_weights_path)
            model = self.load_current_model_state(model, state)
            starting_epoch = 0
            step = 0
            learning_rate = optimizer.defaults["lr"]
            optimizer = optimizer

        elif status == "RESUME":
            resume_weight_path = weight_path
            state = get_current_state(resume_weight_path)
            model = self.load_current_model_state(model, state)
            starting_epoch = state["starting_epoch"] if "starting_epoch" in state else 0
            step = state["step"] if "step" in state else 0
            optimizer.load_state_dict(state["optimizer"])
            learning_rate = optimizer.defaults["lr"]

        elif status == "DEFAULT":
            state = None
            model = self.load_current_model_state(model, state)
            optimizer = optimizer
            starting_epoch = 0
            step = 0
            learning_rate = optimizer.defaults["lr"]

        else:
            raise NotImplementedError
        print(" {} State Loaded with Starting Epoch : {}, Step: {} ".format(status, starting_epoch, step))
        return model, optimizer, learning_rate, starting_epoch, step

    @LogDecorator()
    def load_current_model_state(self, model, state):
        if state is not None:
            model_cuda = adjust_model_keys(state)
            model.load_state_dict(model_cuda)
        model = load_parallel_model(model)
        return model

    @LogDecorator()
    def load_loader(self):
        batch_size = self.training_configuration["initial_assignment"]["batch_size"]
        train_loader = DataLoader(
            dataset=self.plugin.train_data_set,
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            dataset=self.plugin.val_data_set,
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            dataset=self.plugin.test_data_set,
            shuffle=True,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader, test_loader

    @LogDecorator()
    def load_optimizer(self, model):
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
        return optimizer

    @LogDecorator()
    def load_lr_scheduler(self, optimizer):
        if bool(self.training_configuration["scheduler"]):
            lr_scheduler_name = self.training_configuration["scheduler"]
            lr_scheduler_param = self.training_configuration[lr_scheduler_name]

            lr_scheduler = getattr(lr_helper, lr_scheduler_name)(
                **lr_scheduler_param, optimizer=optimizer
            )
        else:
            lr_scheduler = None
            lr_scheduler_name = None
            lr_scheduler_param = None

        return lr_scheduler


