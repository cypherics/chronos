import random
import sys
import os

import numpy as np
import torch

import tqdm

from ml.commons.callbacks import (
    CallbackList,
    TensorBoardCallback,
    TrainStateCallback,
    SchedulerCallback,
    TimeCallback,
)
from ml.commons.utils.torch_tensor_conversion import to_tensor
from ml.commons.utils import torch_tensor_conversion
from ml.state.train import TrainState
from utils.dictionary_set import dict_to_string, handle_dictionary
from ml.pt.logger import PtLogger
from ml.commons.scheduler import get_scheduler


class Trainer(TrainState):
    def __init__(self):
        super().__init__()

    @staticmethod
    @PtLogger()
    def load_optimizer(model, config):
        optimizer_name = config.optimizer
        optimizer_param = config.optimizer_param
        return getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, model.parameters()), **optimizer_param
        )

    @PtLogger()
    def training(self, instance, config, training_callbacks=None):
        self.model = instance.model
        self.optimizer = self.load_optimizer(self.model, config)

        self.extract_state(config.chk_path)
        train_loader, val_loader, test_loader = (
            instance.train_data_loader,
            instance.val_data_loader,
            instance.test_data_loader,
        )

        criterion = instance.criterion
        evaluator = instance.evaluator

        scheduler = get_scheduler(
            config.scheduler,
            **{**config.scheduler_param, **{"optimizer": self.optimizer}}
        )

        callbacks = (
            CallbackList()
            if training_callbacks is None
            else CallbackList(training_callbacks)
        )
        callbacks.append(
            TensorBoardCallback(os.path.join(config.training_path, config.version))
        )
        callbacks.append(TrainStateCallback(config.chk_path, config.best_chk_path))
        callbacks.append(SchedulerCallback(scheduler))
        callbacks.append(TimeCallback())

        report_each = 100

        batch_size = config.batch_size
        epochs = config.n_epochs
        start_epoch = self.starting_epoch

        callbacks.on_begin()

        for ongoing_epoch in range(start_epoch, epochs):
            epoch_logs = dict()
            random.seed()

            self.starting_epoch = ongoing_epoch
            self.model.train()

            tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
            lr = self.optimizer.param_groups[0]["lr"]
            epoch_logs = handle_dictionary(
                epoch_logs, "plt_lr", {"data": lr, "tag": "LR/Epoch"}
            )

            tq.set_description("Epoch {}, lr {}".format(self.starting_epoch, lr))
            losses = []

            tl = train_loader

            try:
                mean_loss = 0
                for i, input_data in enumerate(tl):
                    batch_logs = dict()
                    callbacks.on_batch_begin(self.step, logs=batch_logs)
                    if not self.model.training:
                        self.model.train()

                    input_data = torch_tensor_conversion.cuda_variable(input_data)

                    outputs = self.model(input_data)
                    calculated_loss = criterion(outputs=outputs, **input_data)
                    self.optimizer.zero_grad()
                    calculated_loss.backward()
                    self.optimizer.step()

                    tq.update(batch_size)
                    losses.append(calculated_loss.item())
                    mean_loss = np.mean(losses[-report_each:])
                    tq.set_postfix(loss="{:.5f}".format(mean_loss))
                    batch_logs = handle_dictionary(
                        batch_logs, "plt_lr", {"data": mean_loss, "tag": "Loss/Step"}
                    )
                    if i and i % 20 == 0:
                        save_path = os.path.join(config.training_path, config.version)
                        predicted_images = evaluator.perform_test(
                            self.model, test_loader
                        )
                        evaluator.save_inference_output(
                            predicted_images, save_path, i, ongoing_epoch
                        )
                        batch_logs = handle_dictionary(
                            batch_logs,
                            "plt_img",
                            {
                                "img": to_tensor(np.moveaxis(predicted_images, -1, 0)),
                                "tag": "Test",
                            },
                        )
                    callbacks.on_batch_end(self.step, logs=batch_logs)
                    self.step += 1
                tq.close()

                valid_metrics = evaluator.perform_validation(
                    model=self.model, loss_function=criterion, valid_loader=val_loader
                )
                valid_loss = valid_metrics["valid_loss"]
                epoch_logs = handle_dictionary(epoch_logs, "valid_loss", valid_loss)

                metric_str = dict_to_string(valid_metrics)
                sys.stdout.write("METRIC: {}".format(metric_str))

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

                callbacks.on_epoch_end(
                    self.starting_epoch, logs={**epoch_logs, **self.state_obj}
                )

            except KeyboardInterrupt:
                tq.close()
                callbacks.interruption(logs={**epoch_logs, **self.state_obj})
                sys.stdout.write(
                    "KEYBOARD EXCEPTION CHECKPOINT SAVED : {}".format(ongoing_epoch)
                )
                raise KeyboardInterrupt

            except Exception as ex:
                tq.close()
                raise ex

        sys.stdout.write("Training Complete")
        callbacks.on_end()
