import random
import sys
import os
import time

import numpy as np
import torch

import tqdm

from ml.commons.utils.torch_tensor_conversion import to_tensor
from ml.commons.visualization import Visualization
from ml.commons.utils import torch_tensor_conversion
from utils import date_time_utility
from utils.dictionary_set import dict_to_string
from utils.logger import LogDecorator


class Trainer:
    def __init__(self, plugin, loader, config):
        self.config = config
        self.loader = loader
        self.plugin = plugin

        self.start_time = None
        self.data_viz = None

    @LogDecorator()
    def start_session(self):
        self.start_time = time.time()
        self.data_viz = self.initiate_data_viz()

    @LogDecorator()
    def training(self):
        model = self.plugin.model
        train_loader, val_loader, test_loader = (
            self.plugin.train_data_loader,
            self.plugin.val_data_loader,
            self.plugin.test_data_loader,
        )

        criterion = self.plugin.criterion
        evaluator = self.plugin.evaluator

        optimizer = self.loader.load_optimizer(model)
        scheduler = self.loader.load_lr_scheduler(optimizer)

        model, optimizer, starting_epoch, step, learning_rate = self.loader.load_state(
            model, optimizer, self.config.training_state
        )
        self.start_session()
        report_each = 100
        valid_losses = []
        previous_min_loss = None

        batch_size = self.config.batch_size
        epochs = self.config.n_epochs
        for ongoing_epoch in range(starting_epoch, epochs):
            model.train()
            random.seed()

            tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
            lr = optimizer.param_groups[0]["lr"]
            tq.set_description("Epoch {}, lr {}".format(ongoing_epoch + 1, lr))
            losses = []

            tl = train_loader

            try:
                mean_loss = 0
                for i, input_data in enumerate(tl):

                    if not model.training:
                        model.train()

                    input_data = torch_tensor_conversion.cuda_variable(input_data)

                    outputs = model(input_data)
                    calculated_loss = criterion(outputs=outputs, **input_data)
                    optimizer.zero_grad()
                    # batch_size = inputs.size(0)
                    calculated_loss.backward()
                    optimizer.step()

                    tq.update(batch_size)
                    losses.append(calculated_loss.item())
                    mean_loss = np.mean(
                        losses[-report_each:]
                    )  # understand pythonic interpretation of this line
                    tq.set_postfix(loss="{:.5f}".format(mean_loss))
                    self.data_viz.plt_scalar(mean_loss.item(), step, "Loss/Step")
                    step += 1
                    if i and i % 20 == 0:
                        save_path = os.path.join(
                            self.config.training_path, self.config.version
                        )
                        predicted_images = evaluator.perform_test(model, test_loader)
                        evaluator.save_inference_output(
                            predicted_images, save_path, i, ongoing_epoch
                        )
                        self.data_viz.plt_images(
                            to_tensor(np.moveaxis(predicted_images, -1, 0)),
                            ongoing_epoch,
                            "Test",
                        )

                #
                self.data_viz.plt_scalar(lr, ongoing_epoch, "LR/Epoch")

                tq.close()

                valid_metrics = evaluator.perform_validation(
                    model=model, loss_function=criterion, valid_loader=val_loader
                )
                valid_loss = valid_metrics["valid_loss"]
                valid_losses.append(valid_loss)

                if scheduler is not None:
                    scheduler.step(epoch=ongoing_epoch + 1, valid_loss=valid_loss)

                metric_str = dict_to_string(valid_metrics)
                sys.stdout.write("METRIC: {}".format(metric_str))

                # self.early_stopping.step(valid_loss)

                self.data_viz.plt_scalar(
                    {"train_loss": mean_loss.item(), "val_loss": valid_loss},
                    ongoing_epoch,
                    "Loss/Epoch",
                )

                if (previous_min_loss is None) or (valid_loss < previous_min_loss):
                    previous_min_loss = valid_loss

                    self.save_check_point(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        epoch=ongoing_epoch,
                        lr=lr,
                        save_type="best",
                    )
                    sys.stdout.write(
                        "BEST CHECKPOINT SAVED at Epoch: {}".format(ongoing_epoch)
                    )

                self.save_check_point(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=ongoing_epoch,
                    lr=lr,
                )
                sys.stdout.write(
                    "DEFAULT CHECKPOINT SAVED at Epoch: {}".format(ongoing_epoch)
                )

            except KeyboardInterrupt:
                tq.close()
                self.close_session()
                self.save_check_point(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=ongoing_epoch,
                    lr=lr,
                )
                sys.stdout.write(
                    "KEYBOARD EXCEPTION CHECKPOINT SAVED : {}".format(ongoing_epoch)
                )

                raise KeyboardInterrupt

            except Exception as ex:
                tq.close()
                self.close_session()
                raise ex

        self.close_session()
        sys.stdout.write("Training Complete")

    @LogDecorator()
    def initiate_data_viz(self):
        log_dir = os.path.join(self.config.training_path, self.config.version)
        data_viz = Visualization(log_dir)
        return data_viz

    @LogDecorator()
    def close_session(self):
        end_time = time.time()
        total_time = date_time_utility.get_time(end_time - self.start_time)
        sys.stdout.write("Run Time : {}".format(total_time))

    def best(self):
        return self.config.best_chk_path

    def default(self):
        return self.config.chk_path

    def save_check_point(self, model, optimizer, epoch, step, lr, save_type="default"):
        func = getattr(self, save_type)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "starting_epoch": epoch,
                "step": step,
                "lr": lr,
            },
            str(func()),
        )
