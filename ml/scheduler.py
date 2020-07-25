import math
import sys

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from core.logger import debug, ChronosLogger

logger = ChronosLogger.get_logger()


class PolyLrDecay(_LRScheduler):
    @debug
    def __init__(self, power, max_epochs, optimizer, epoch):
        self.max_epoch = max_epochs
        self.power = power
        if epoch == 1:
            start_epoch = -1
        else:
            start_epoch = epoch - 1
        super(PolyLrDecay, self).__init__(optimizer, start_epoch)

    def get_lr(self):

        new_lrs = [
            base_lr * (1 - (self.last_epoch - 1) / self.max_epoch) ** self.power
            for base_lr in self.base_lrs
        ]
        logger.debug("New learning Rate Set {}".format(new_lrs))
        return new_lrs

    def step(self, epoch):
        self.last_epoch = epoch + 1
        logger.debug("Scheduler Taking Step {}".format(self.last_epoch))
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def lr_finder_lambda(end_lr, start_lr, total_epoch, tran_loader_len):
    lr_lambda = lambda x: math.exp(
        x * math.log(end_lr / start_lr) / (total_epoch * tran_loader_len)
    )
    return lr_lambda


@debug
def get_scheduler(scheduler: str, **kwargs):
    if hasattr(lr_scheduler, scheduler):
        return getattr(lr_scheduler, scheduler)(**kwargs)
    else:
        return str_to_class(scheduler)(**kwargs)


def str_to_class(class_name: str):
    class_obj = getattr(sys.modules[__name__], class_name)
    return class_obj
