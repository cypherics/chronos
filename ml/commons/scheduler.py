import sys

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from ml.pt.logger import PtLogger


class PolyLrDecay(_LRScheduler):
    # the calculation here is done from epoch == 0 instead of 1
    def __init__(self, decay_after_epoch, power, max_epochs, optimizer, epoch):
        self.decay_after_epoch = decay_after_epoch
        self.max_epoch = max_epochs
        self.power = power
        if epoch == 1:
            start_epoch = -1
        else:
            start_epoch = epoch
        super(PolyLrDecay, self).__init__(optimizer, start_epoch)

    def get_lr(self):
        new_lrs = [
            base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power
            for base_lr in self.base_lrs
        ]

        return new_lrs

    def step(self, epoch):
        self.last_epoch = epoch - 1 if epoch != 0 else epoch
        if epoch % self.decay_after_epoch == 0:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr


@PtLogger(debug=True)
def get_scheduler(scheduler: str, **kwargs):
    if hasattr(lr_scheduler, scheduler):
        return getattr(lr_scheduler, scheduler)(**kwargs)
    else:
        return str_to_class(scheduler)(**kwargs)


@PtLogger(debug=True)
def str_to_class(class_name: str):
    class_obj = getattr(sys.modules[__name__], class_name)
    return class_obj
