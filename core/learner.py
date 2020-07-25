import random

import numpy as np
import torch

import tqdm

from core.extensions.callbacks import CallbackList, SchedulerCallback
from core.extensions.metric import MetricList
from utils import pt_tensor
from core.state import LearnerState
from utils.dict_ops import dict_to_string, handle_dictionary
from core.logger import info, ChronosLogger
from ml.scheduler import get_scheduler
from utils.system_printer import SystemPrinter

logger = ChronosLogger.get_logger()


class Learner(LearnerState):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @info
    def training(
        self, plugin, optimizer, training_callbacks: CallbackList, metrics: MetricList
    ):

        batch_size = self.config.batch_size
        epochs = self.config.n_epochs
        self.restart(
            plugin.model, optimizer, self.config.default_state
        ) if self.config.resume else self.new(plugin.model, optimizer)

        if self.config.scheduler_name is not None:
            scheduler = get_scheduler(
                self.config.scheduler_name,
                **{
                    **self.config.scheduler_param,
                    **{"optimizer": self.optimizer, "epoch": self.starting_epoch},
                }
            )

            training_callbacks.append(SchedulerCallback(scheduler))
        training_callbacks.on_begin()

        begin_epoch = self.starting_epoch
        for ongoing_epoch in range(begin_epoch, epochs + 1):
            epoch_logs = dict()
            random.seed()

            self.starting_epoch = ongoing_epoch
            self.model.train()
            lr = self.optimizer.param_groups[0]["lr"]

            progress_bar = tqdm.tqdm(total=(len(plugin.loader.train_data) * batch_size))
            progress_bar.set_description(
                "Epoch {}, lr {}".format(self.starting_epoch, lr)
            )

            try:
                logger.debug("Setting Learning rate : {}".format(lr))
                epoch_logs = handle_dictionary(epoch_logs, "lr", lr)

                train_loss, train_metric, progress_bar = self.state_train(
                    plugin, training_callbacks, batch_size, metrics, progress_bar
                )
                progress_bar.close()

                valid_loss, valid_metric = self.state_validate(plugin, metrics)

                epoch_logs = handle_dictionary(epoch_logs, "train_loss", train_loss)
                epoch_logs = handle_dictionary(epoch_logs, "valid_loss", valid_loss)

                epoch_logs = handle_dictionary(epoch_logs, "train_metric", train_metric)
                epoch_logs = handle_dictionary(epoch_logs, "valid_metric", valid_metric)

                if (self.bst_vld_loss is None) or (valid_loss < self.bst_vld_loss):
                    self.bst_vld_loss = valid_loss

                epoch_logs = handle_dictionary(epoch_logs, "model", self.model)
                epoch_logs = handle_dictionary(
                    epoch_logs, "test_loader", plugin.loader.test_data
                )

                training_callbacks.on_epoch_end(
                    self.starting_epoch, logs={**epoch_logs, **self.epoch_state}
                )

                logger.debug(
                    "Train Loss {}, Valid Loss {}".format(train_loss, valid_loss)
                )
                logger.debug("Train Metric {}".format(train_metric))
                logger.debug("Valid Metric {}".format(valid_metric))

                SystemPrinter.sys_print(
                    "Epoch: {}, TrainLoss: {}, ValidLoss: {}".format(
                        ongoing_epoch, train_loss, valid_loss
                    )
                )
                SystemPrinter.sys_print(
                    "Train Metric: {}".format(dict_to_string(train_metric))
                )
                SystemPrinter.sys_print(
                    "Valid Metric: {}".format(dict_to_string(valid_metric))
                )

            except KeyboardInterrupt:
                progress_bar.close()
                training_callbacks.interruption(
                    logs={**epoch_logs, **self.interruption_state}
                )
                SystemPrinter.sys_print(
                    "KEYBOARD EXCEPTION CHECKPOINT SAVED : {}".format(ongoing_epoch)
                )
                raise KeyboardInterrupt

            except Exception as ex:
                progress_bar.close()
                raise ex

        SystemPrinter.sys_print("Training Complete")
        training_callbacks.on_end()

    def state_train(self, plugin, callbacks, batch_size, metrics, progress_bar):

        report_each = 100
        batch_loss = []
        mean_loss = 0
        for images, ground_truth in plugin.loader.train_data:
            batch_logs = dict()
            callbacks.on_batch_begin(self.step, logs=batch_logs)
            if not self.model.training:
                self.model.train()

            images = pt_tensor.make_cuda(images)
            ground_truth = pt_tensor.make_cuda(ground_truth)

            prediction = self.model(images)
            assert type(prediction) == dict, "Model Must Return A Dict"
            calculated_loss = plugin.criterion(ground_truth, prediction)
            self.optimizer.zero_grad()
            calculated_loss.backward()
            self.optimizer.step()

            batch_loss.append(calculated_loss.item())
            mean_loss = np.mean(batch_loss[-report_each:])
            batch_logs = handle_dictionary(
                batch_logs, "plt_lr", {"data": mean_loss, "tag": "Loss/Step"}
            )
            batch_logs = handle_dictionary(batch_logs, "model", self.model)
            batch_logs = handle_dictionary(
                batch_logs, "test_loader", plugin.loader.test_data
            )
            callbacks.on_batch_end(self.step, logs=batch_logs)
            progress_bar.update(batch_size)
            progress_bar.set_postfix(loss="{:.5f}".format(mean_loss))
            self.step += 1
            metrics.get_metrics(ground_truth=ground_truth, prediction=prediction)
        return mean_loss.item(), metrics.compute_mean(), progress_bar

    @torch.no_grad()
    def state_validate(self, plugin, metrics):
        logger.debug("Validation In Progress")
        self.model.eval()
        losses = []
        ongoing_count = 1
        total_count = len(plugin.loader.val_data)
        sys_print = SystemPrinter()
        for images, ground_truth in plugin.loader.val_data:
            sys_print.dynamic_print(
                tag=str("Validation"),
                data="{}/{} -> {}".format(
                    ongoing_count,
                    total_count,
                    sys_print.compute_eta(ongoing_count, total_count),
                ),
            )

            ongoing_count += 1
            images = pt_tensor.make_cuda(images)
            ground_truth = pt_tensor.make_cuda(ground_truth)

            prediction = self.model(images)
            loss = plugin.criterion(ground_truth, prediction)

            losses.append(loss.item())
            metrics.get_metrics(ground_truth=ground_truth, prediction=prediction)

        valid_loss = np.mean(losses)
        return valid_loss, metrics.compute_mean()
