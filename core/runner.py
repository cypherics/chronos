import random

import numpy as np
import torch

import tqdm

from core.callbacks import CallbackList, SchedulerCallback
from core.data.metrics import compute_metric, compute_mean_metric
from utils import pt_tensor
from utils.network_util import get_prediction_as_per_instance
from utils.pt_tensor import make_cuda
from core.state import PtState
from utils.dict_ops import dict_to_string, handle_dictionary
from core.logger import info, ChronosLogger
from core.scheduler import get_scheduler
from utils.system_printer import SystemPrinter

logger = ChronosLogger.get_logger()


class PtRunner(PtState):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @info
    def training(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        training_callbacks: CallbackList,
    ):

        batch_size = self.config.batch_size
        epochs = self.config.n_epochs
        self.restart(
            model, optimizer, self.config.default_state
        ) if self.config.resume else self.new(model, optimizer)

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
        for ongoing_epoch in range(begin_epoch, epochs):
            epoch_logs = dict()
            random.seed()

            self.starting_epoch = ongoing_epoch
            self.model.train()
            lr = self.optimizer.param_groups[0]["lr"]

            progress_bar = tqdm.tqdm(total=(len(train_loader) * batch_size))
            progress_bar.set_description(
                "Epoch {}, lr {}".format(self.starting_epoch, lr)
            )

            try:
                logger.debug("Setting Learning rate : {}".format(lr))
                epoch_logs = handle_dictionary(
                    epoch_logs, "plt_lr", {"data": lr, "tag": "LR/Epoch"}
                )

                mean_loss, progress_bar = self.state_train(
                    train_loader,
                    criterion,
                    training_callbacks,
                    batch_size,
                    progress_bar,
                )
                progress_bar.close()

                valid_metrics = self.state_validate(criterion, val_loader)
                valid_loss = valid_metrics["valid_loss"]

                epoch_logs = handle_dictionary(epoch_logs, "valid_loss", valid_loss)
                logger.debug(
                    "Train Loss {}, Valid Loss {}".format(mean_loss, valid_loss)
                )
                logger.debug("Metric {}, Chk Saved".format(valid_metrics))

                metric_str = dict_to_string(
                    {
                        **{"Epoch": ongoing_epoch, "train_loss": mean_loss},
                        **valid_metrics,
                    }
                )
                SystemPrinter.sys_print("{}".format(metric_str))
                epoch_logs = handle_dictionary(
                    epoch_logs,
                    "plt_loss",
                    {
                        "data": {
                            "train_loss": mean_loss.item(),
                            "val_loss": valid_loss,
                        },
                        "tag": "Loss/Epoch",
                    },
                )

                if (self.bst_vld_loss is None) or (valid_loss < self.bst_vld_loss):
                    self.bst_vld_loss = valid_loss

                training_callbacks.on_epoch_end(
                    self.starting_epoch, logs={**epoch_logs, **self.epoch_state}
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

    def state_train(self, train_loader, criterion, callbacks, batch_size, progress_bar):
        report_each = 100
        batch_loss = []
        mean_loss = 0
        for batch_iterator, input_data in enumerate(train_loader):
            batch_logs = dict()
            callbacks.on_batch_begin(self.step, logs=batch_logs)
            if not self.model.training:
                self.model.train()

            input_data = pt_tensor.make_cuda(input_data)

            outputs = self.model(input_data)
            calculated_loss = criterion(outputs=outputs, **input_data)
            self.optimizer.zero_grad()
            calculated_loss.backward()
            self.optimizer.step()

            batch_loss.append(calculated_loss.item())
            mean_loss = np.mean(batch_loss[-report_each:])
            batch_logs = handle_dictionary(
                batch_logs, "plt_lr", {"data": mean_loss, "tag": "Loss/Step"}
            )
            batch_logs = handle_dictionary(batch_logs, "model", self.model)
            callbacks.on_batch_end(self.step, logs=batch_logs)
            progress_bar.update(batch_size)
            progress_bar.set_postfix(loss="{:.5f}".format(mean_loss))
            self.step += 1
        return mean_loss, progress_bar

    @torch.no_grad()
    def state_validate(self, criterion, valid_loader):
        logger.debug("Validation In Progress")
        self.model.eval()
        losses = []
        metric = dict()
        ongoing_count = 1
        total_count = len(valid_loader)
        sys_print = SystemPrinter()
        for input_data in valid_loader:
            sys_print.dynamic_print(
                tag=str("Validation"),
                data="{}/{} -> {}".format(
                    ongoing_count,
                    total_count,
                    sys_print.compute_eta(ongoing_count, total_count),
                ),
            )

            ongoing_count += 1
            input_data = make_cuda(input_data)

            targets = input_data["label"]

            input_data = make_cuda(input_data)
            outputs = self.model(input_data)
            loss = criterion(outputs, **input_data)

            losses.append(loss.item())
            outputs = get_prediction_as_per_instance(outputs)

            met = compute_metric(ground_truth=targets, prediction=outputs)
            if met is not None:
                for key, value in met.items():
                    metric = handle_dictionary(metric, key, value)

        valid_loss = np.mean(losses)
        valid_loss = {"valid_loss": valid_loss}

        validation_metric = compute_mean_metric(metric)
        metrics = {**valid_loss, **validation_metric}
        return metrics
