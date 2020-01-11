import numpy as np
import math

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler


class ReduceOnPlateau:
    def __init__(self, optimizer, patience=6, factor=0.7):
        self.patience = patience
        self.factor = factor
        self.optimizer = optimizer

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=self.patience,
            verbose=True,
            factor=self.factor,
        )

    def step(self, **kwargs):
        valid_loss = kwargs["valid_loss"]
        self.scheduler.step(valid_loss)


class ExponentialLr(_LRScheduler):
    def __init__(self, decay_after_epoch, gamma, optimizer, last_epoch=-1):
        self.decay_after_epoch = decay_after_epoch
        self.gamma = gamma
        self.optimizer = optimizer
        self.current_epoch = last_epoch

        self.lr_history = []

        super(ExponentialLr, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]

    def step(self, **kwargs):
        epoch = kwargs["epoch"]
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch = self.last_epoch

        if (epoch + 1) % self.decay_after_epoch == 0:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr

        else:
            return self.optimizer


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


class SGDRWithRestart(_LRScheduler):
    def __init__(
        self,
        max_num_iter,
        multiplying_factor,
        min_lr_rate,
        out_dir,
        take_snapshot,
        optimizer,
        last_epoch=-1,
    ):
        self.max_num_iter = max_num_iter
        self.multiplying_factor = multiplying_factor
        self.restart_at = self.max_num_iter
        self.eta_min = min_lr_rate
        self.current_epoch = 0

        self.out_dir = out_dir
        self.take_snapshot = take_snapshot

        self.lr_history = []

        # self.optimizer = optimizer
        super(SGDRWithRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.current_epoch / self.restart_at))
            / 2
            for base_lr in self.base_lrs
        ]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, **kwargs):
        epoch = kwargs["epoch"]
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch = self.last_epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        if self.current_epoch == self.restart_at:
            print("restart at starting_epoch {:03d}".format(self.last_epoch + 1))

            # TODO write save snapshot

            self.current_epoch = 0

            self.restart_at = int(self.restart_at * self.multiplying_factor)
            self.max_num_iter = self.max_num_iter + self.restart_at


class PolyLrDecay(_LRScheduler):
    def __init__(self, decay_after_epoch, power, max_epochs, optimizer, last_epoch=-1):
        self.decay_after_epoch = decay_after_epoch
        self.max_epoch = max_epochs
        self.power = power
        self.current_epoch = last_epoch

        self.lr_history = []

        super(PolyLrDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [
            base_lr * (1 - self.current_epoch / self.max_epoch) ** self.power
            for base_lr in self.base_lrs
        ]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None, valid_loss=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch = self.last_epoch

        if (epoch + 1) % self.decay_after_epoch == 0:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr

        else:
            return self.optimizer


# for i in range(0, 200):
#     print("{} : {}".format(i + 1, 1e-4 * 0.90 ** i))

# for i in range(0, 120):
#     print("{} : {}".format(i + 1, 1e-04 * (1 - i / 120) ** 0.90))
