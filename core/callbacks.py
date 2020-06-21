import os
import random
import shutil

import torch
import warnings
import time

import cv2

from core.utils.model_util import adjust_model, get_prediction_as_per_instance
from core.logger import debug, ChronosLogger
from core.utils.tensor_util import cuda_variable
from utils import date_time_utility
from utils.dictionary_set import handle_dictionary
from utils.directory_handler import make_directory
from utils.function_util import is_overridden_func
from utils.system_printer import SystemPrinter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from torch.utils.tensorboard import SummaryWriter

logger = ChronosLogger.get_logger()


class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        if len(callbacks) != 0:
            [
                logger.debug("Registered {}".format(c.__class__.__name__))
                for c in callbacks
            ]

    def append(self, callback):
        logger.debug("Registered {}".format(callback.__class__.__name__))
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_begin):
                logger.debug(
                    "Nothing Registered On Epoch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_end):
                logger.debug(
                    "Nothing Registered On Epoch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            logger.debug("On Batch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_begin):
                logger.debug(
                    "Nothing Registered On Batch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):

        for callback in self.callbacks:
            logger.debug("On Batch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_end):
                logger.debug(
                    "Nothing Registered On Batch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_end(batch, logs)

    def on_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_begin):
                logger.debug(
                    "Nothing Registered On Begin {}".format(callback.__class__.__name__)
                )
            callback.on_begin(logs)

    def on_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_end):
                logger.debug(
                    "Nothing Registered On End {}".format(callback.__class__.__name__)
                )
            callback.on_end(logs)

    def interruption(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("Interruption {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.interruption):
                logger.debug(
                    "Nothing Registered On Interruption {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.interruption(logs)

    def update_params(self, params):
        for callback in self.callbacks:
            if not is_overridden_func(callback.update_params):
                logger.debug(
                    "Nothing Registered On Update param {}".format(
                        callback.__class__.__name__
                    )
                )
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


class TrainStateCallback(Callback):
    @debug
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
        logger.debug(
            "Successful on Epoch End {}, Saved State".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["my_state"]

        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Interruption {}, Saved State".format(self.__class__.__name__)
        )


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
        logger.debug(
            "Successful on Epoch End {}, Data Plot".format(self.__class__.__name__)
        )

    def on_batch_end(self, batch, logs=None):
        img_data = logs["plt_img"] if "plt_img" in logs else None
        data = logs["plt_lr"]

        if img_data is not None:
            # self.plt_images(to_tensor(np.moveaxis(img_data["img"], -1, 0)), batch, img_data["tag"])
            pass

        self.plt_scalar(data["data"], batch, data["tag"])
        logger.debug(
            "Successful on Batch End {}, Data Plot".format(self.__class__.__name__)
        )


class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(epoch)
        logger.debug(
            "Successful on Epoch End {}, Lr Scheduled".format(self.__class__.__name__)
        )


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
            cv2.imwrite(
                save_image_path, cv2.cvtColor(img_data["img"], cv2.COLOR_RGB2BGR)
            )
            logger.debug(
                "Successful on Batch End {}, Images Saved".format(
                    self.__class__.__name__
                )
            )


class TrainChkCallback(Callback):
    @debug
    def __init__(self, save_path):
        super().__init__()
        self.chk = save_path

    def on_epoch_end(self, epoch, logs=None):
        my_state = logs["my_state"]
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Chk Saved".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["my_state"]
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on interruption {}, Chk Saved".format(self.__class__.__name__)
        )


class TestCallback(Callback):
    @debug
    def __init__(self, test_loader, evaluator, pth):
        super().__init__()
        self.test_loader = test_loader
        self.evaluator = evaluator
        self.prediction_save_callback = PredictionSaveCallback(pth)

    def on_batch_end(self, batch, logs=None):
        model = logs["model"]
        model.eval()
        if random.random() < 0.10:
            try:
                for i, (inputs, file_path) in enumerate(self.test_loader):

                    image = cuda_variable(inputs)
                    prediction = model(image)
                    prediction = get_prediction_as_per_instance(prediction)
                    prediction = self.evaluator.handle_prediction(prediction)
                    prediction = self.evaluator.create_prediction_grid(
                        inputs, prediction
                    )
                    logs = handle_dictionary(
                        logs, "plt_img", {"img": prediction, "tag": "Test"}
                    )
                    self.prediction_save_callback.on_batch_end(batch, logs)
                    break
            except Exception as ex:
                logger.exception(
                    "Skipped Exception in {}".format(self.__class__.__name__)
                )
                logger.exception("Exception {}".format(ex))
                pass
