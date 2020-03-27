import os
import shutil

import torch
import warnings
import time

import numpy as np
import cv2
from ml.commons.utils.torch_tensor_conversion import to_tensor
from ml.pt.logger import PtLogger
from utils import date_time_utility
from utils.directory_handler import make_directory
from utils.system_printer import SystemPrinter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from torch.utils.tensorboard import SummaryWriter


class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):

        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_begin(logs)

    def on_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_end(logs)

    def interruption(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.interruption(logs)

    def update_params(self, params):
        for callback in self.callbacks:
            callback.update_params(params)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    def __init__(self):
        self.validation_data = None

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_begin(self, logs=None):
        pass

    def on_end(self, logs=None):
        pass

    def interruption(self, logs=None):
        pass

    def update_params(self, params):
        pass


@PtLogger(debug=True)
class TrainStateCallback(Callback):
    def __init__(self, save_path, best_save_path):
        super().__init__()
        self.chk = save_path
        self.best = best_save_path
        self.previous_best = None

    def on_epoch_end(self, epoch, logs=None):
        valid_loss = logs["valid_loss"]
        my_state = logs["my_state"]
        if self.previous_best is None or valid_loss < self.previous_best:
            self.previous_best = valid_loss
            torch.save(my_state, str(self.best))
        torch.save(my_state, str(self.chk))

    def interruption(self, logs=None):
        my_state = logs["my_state"]

        torch.save(my_state, str(self.chk))


class TensorBoardCallback(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(make_directory(log_dir, "events"))

    def plt_scalar(self, y, x, tag):
        if type(y) is dict:
            self.writer.add_scalars(tag, y, global_step=x)
            self.writer.flush()
        else:
            self.writer.add_scalar(tag, y, global_step=x)
            self.writer.flush()

    def plt_images(self, img, global_step, tag):
        self.writer.add_image(tag, img, global_step)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        loss = logs["plt_loss"]
        lr = logs["plt_lr"]
        self.plt_scalar(lr["data"], epoch, lr["tag"])
        self.plt_scalar(loss["data"], epoch, loss["tag"])

    def on_batch_end(self, batch, logs=None):
        img_data = logs["plt_img"] if "plt_img" in logs else None
        data = logs["plt_lr"]

        if img_data is not None:
            # self.plt_images(to_tensor(np.moveaxis(img_data["img"], -1, 0)), batch, img_data["tag"])
            pass

        self.plt_scalar(data["data"], batch, data["tag"])


class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(epoch)


class TimeCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_begin(self, logs=None):
        self.start_time = time.time()

    def on_end(self, logs=None):
        end_time = time.time()
        total_time = date_time_utility.get_time(end_time - self.start_time)
        SystemPrinter.sys_print("Run Time : {}".format(total_time))

    def interruption(self, logs=None):
        end_time = time.time()
        total_time = date_time_utility.get_time(end_time - self.start_time)
        SystemPrinter.sys_print("Run Time : {}".format(total_time))


class PredictionSaveCallback(Callback):
    def __init__(self, pth):
        super().__init__()
        self.save_path = pth

    def on_batch_end(self, batch, logs=None):
        img_data = logs["plt_img"] if "plt_img" in logs else None

        if img_data is not None:
            save_path = make_directory(self.save_path, "test_prediction")

            shutil.rmtree(save_path)
            os.makedirs(save_path)

            save_image_path = os.path.join(save_path, "{}.png".format(batch))
            cv2.imwrite(save_image_path, img_data["img"])
