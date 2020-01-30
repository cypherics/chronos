import math
import random
import sys
import time
import traceback

import numpy as np
import torch
import tqdm

from utils import date_time_utility, print_format
from ml.commons.utils import torch_tensor_conversion

from ml.commons.data_visualization import DataViz


class LrFinderPt:
    def __init__(self, loader_pt, model_pt, lr_finder_configuration, logger):
        self.lr_finder_configuration = lr_finder_configuration
        self.logger = logger

        self.loader_pt = loader_pt
        self.model_pt = model_pt

    @staticmethod
    def schedule_optimizer(
        optimizer, end_lr, start_lr, number_of_epochs, len_of_train_data
    ):
        lr_lambda = lambda x: math.exp(
            x * math.log(end_lr / start_lr) / (number_of_epochs * len_of_train_data)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler

    def find_lr(self):
        start_time = time.time()
        model, optimizer, loss, lr_scheduler, learning_rate, starting_epoch, step = (
            self.initialize_required_training_param()
        )
        lr_scheduler = self.schedule_optimizer(
            optimizer,
            self.lr_finder_configuration["initial_assignment"]["end_lr"],
            self.lr_finder_configuration["initial_assignment"]["start_lr"],
            self.lr_finder_configuration["initial_assignment"]["n_epochs"],
            len(self.loader_pt.train_loader),
        )
        data_viz = self.initiate_data_viz()

        print_format.colored_dual_string_print_with_brackets(
            str("DEFAULT"), "LR-Finder", "red", "cyan", attrs=["blink"]
        )

        report_each = 100

        lr_find_lr = []
        lr_find_loss = []
        smoothing = 0.05

        batch_size = self.lr_finder_configuration["initial_assignment"]["batch_size"]

        for ongoing_epoch in range(
            starting_epoch,
            self.lr_finder_configuration["initial_assignment"]["n_epochs"],
        ):
            model.train()
            random.seed()

            tq = tqdm.tqdm(total=(len(self.loader_pt.train_loader) * batch_size))
            tq.set_description("Epoch {}".format(ongoing_epoch + 1))
            losses = []

            tl = self.loader_pt.train_loader

            try:
                for i, input_data in enumerate(tl):

                    if not model.training:
                        model.train()

                    input_data = torch_tensor_conversion.cuda_variable(input_data)

                    outputs = model(input_data)
                    calculated_loss = loss(outputs=outputs, **input_data)
                    optimizer.zero_grad()
                    calculated_loss.backward()
                    optimizer.step()

                    tq.update(batch_size)
                    current_loss = calculated_loss.item()
                    losses.append(current_loss)
                    mean_loss = np.mean(losses[-report_each:])
                    tq.set_postfix(loss="{:.5f}".format(mean_loss))
                    lr_scheduler.step()
                    lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
                    lr_find_lr.append(lr_step)
                    if step == 0:
                        lr_find_loss.append(current_loss)
                    else:
                        current_loss = (
                            smoothing * current_loss
                            + (1 - smoothing) * lr_find_loss[-1]
                        )
                        lr_find_loss.append(current_loss)
                    step += 1
                    data_viz.plot_lr_loss(lr_step, current_loss)
                    self.logger.log_info(
                        "LR: {}, Loss: {}".format(lr_step, current_loss)
                    )
                tq.close()

            except KeyboardInterrupt:
                tq.close()
                data_viz.save_viz()
                data_viz.terminate_viz()

                end_time = time.time()
                total_time = date_time_utility.get_time(end_time - start_time)
                self.logger.log_exception(KeyboardInterrupt)
                print_format.colored_dual_string_print_with_brackets(
                    "Run Time", total_time, "green", "yellow", attrs=["bold", "blink"]
                )
                sys.exit(str(traceback.format_exc()))

            except Exception as ex:
                data_viz.save_viz()
                data_viz.terminate_viz()
                end_time = time.time()
                total_time = date_time_utility.get_time(end_time - start_time)
                self.logger.log_exception(ex)
                self.logger.log_info("Total Time : {}".format(str(total_time)))
                print("\n")

                print_format.colored_dual_string_print_with_brackets(
                    "Run Time", total_time, "green", "yellow", attrs=["bold", "blink"]
                )
                sys.exit(str(traceback.format_exc()))

        data_viz.save_viz()
        data_viz.terminate_viz()
        end_time = time.time()
        total_time = date_time_utility.get_time(end_time - start_time)
        self.logger.log_info("Total Time : {}".format(str(total_time)))
        print_format.colored_dual_string_print_with_brackets(
            "DONE-LRFinder", total_time, "green", "yellow", attrs=["bold", "blink"]
        )

    def initialize_required_training_param(self):
        optimizer = self.loader_pt.load_optimizer(self.model_pt.model)
        loss = self.loader_pt.load_loss()
        lr_scheduler = self.loader_pt.load_lr_scheduler(optimizer)
        model, optimizer, learning_rate, starting_epoch, step = self.model_pt.load_model_weights_and_params(
            optimizer
        )
        return model, optimizer, loss, lr_scheduler, learning_rate, starting_epoch, step

    def initiate_data_viz(self):
        data_viz = DataViz(
            env_name=str(
                self.lr_finder_configuration["experiment_name"]
                + str("_")
                + self.lr_finder_configuration["model"]
                + str("_")
                + str("lr_finder")
            ),
            logger=self.logger,
            find_lr=True,
        )
        return data_viz
