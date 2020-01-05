import os
import random
import sys
import time
import traceback

import numpy as np
import tqdm

from utils import date_time_utility, yaml_utility, print_format, torch_tensor_conversion
from utils.directory_utils import path_creation

from utils.visdom_data_visualization import DataViz


class Trainer:
    def __init__(self, parameter_loader, training_configuration, logger, training_path):
        self.training_configuration = training_configuration
        self.logger = logger
        self.training_path = training_path

        self.parameter_loader = parameter_loader
        self.status, self.version = self.parameter_loader.get_model_version_and_status(
            self.training_path
        )
        self.default_weights_path, self.best_weights_path = path_creation.create_weights_path(
            self.training_path, self.version
        )

        # self.model = self.model_loader.model

    def training(self):
        start_time = time.time()
        model, optimizer, loss, lr_scheduler, learning_rate, starting_epoch, step = (
            self.initialize_required_training_param()
        )

        data_viz = self.initiate_data_viz()

        print_format.colored_dual_string_print_with_brackets(
            "Run Path",
            "{}/{}".format(self.training_path, self.version),
            "red",
            "yellow",
        )
        print_format.colored_dual_string_print_with_brackets(
            str(self.status), "Training", "red", "cyan", attrs=["blink"]
        )

        report_each = 100
        valid_losses = []
        previous_min_loss = None

        batch_size = self.training_configuration["initial_assignment"]["batch_size"]

        for ongoing_epoch in range(
            starting_epoch,
            self.training_configuration["initial_assignment"]["n_epochs"],
        ):
            model.train()
            random.seed()

            tq = tqdm.tqdm(total=(len(self.parameter_loader.train_loader) * batch_size))
            lr = optimizer.param_groups[0]["lr"]
            tq.set_description("Epoch {}, lr {}".format(ongoing_epoch + 1, lr))
            losses = []

            tl = self.parameter_loader.train_loader

            try:
                mean_loss = 0
                for i, input_data in enumerate(tl):

                    if not model.training:
                        model.train()

                    input_data = torch_tensor_conversion.cuda_variable(input_data)

                    outputs = model(input_data)
                    calculated_loss = loss(outputs=outputs, **input_data)
                    optimizer.zero_grad()
                    # batch_size = inputs.size(0)
                    calculated_loss.backward()
                    optimizer.step()

                    step += 1
                    tq.update(batch_size)
                    losses.append(calculated_loss.item())
                    mean_loss = np.mean(
                        losses[-report_each:]
                    )  # understand pythonic interpretation of this line
                    tq.set_postfix(loss="{:.5f}".format(mean_loss))

                    if i and i % 20 == 0:
                        predicted_images = self.parameter_loader.validation.inference(
                            model,
                            i,
                            ongoing_epoch,
                            self.parameter_loader.test_loader,
                            save_path=os.path.join(self.training_path, self.version),
                        )
                        data_viz.plot_test_image(predicted_images)
                #
                data_viz.plot_learning_rate(lr, ongoing_epoch)
                #
                data_viz.plot_train_loss(mean_loss, ongoing_epoch)

                tq.close()

                valid_metrics = self.parameter_loader.validation.perform_validation(
                    model=model,
                    loss_function=loss,
                    valid_loader=self.parameter_loader.val_loader,
                )
                valid_loss = valid_metrics["valid_loss"]
                valid_losses.append(valid_loss)

                if lr_scheduler is not None:
                    lr_scheduler.step(epoch=ongoing_epoch + 1, valid_loss=valid_loss)

                # self.early_stopping.step(valid_loss)

                data_viz.plot_val_loss(valid_loss, ongoing_epoch)

                if (previous_min_loss is None) or (valid_loss < previous_min_loss):
                    previous_min_loss = valid_loss

                    self.parameter_loader.save_check_point(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        epoch=ongoing_epoch,
                        lr=lr,
                        model_path=self.best_weights_path,
                    )
                    self.logger.log_info(
                        "Best Weights Saved | epoch :  {} train loss : {} : validation metric : {}".format(
                            ongoing_epoch, mean_loss, valid_metrics
                        )
                    )
                self.parameter_loader.save_check_point(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=ongoing_epoch,
                    lr=lr,
                    model_path=self.default_weights_path,
                )
                self.logger.log_info(
                    "Default Weights Saved | epoch :  {} train loss : {} : validation metric : {}".format(
                        ongoing_epoch, mean_loss, valid_metrics
                    )
                )

            except KeyboardInterrupt:
                tq.close()
                data_viz.save_viz()
                data_viz.terminate_viz()

                end_time = time.time()
                total_time = date_time_utility.get_time(end_time - start_time)
                self.parameter_loader.save_check_point(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=ongoing_epoch,
                    lr=lr,
                    model_path=self.default_weights_path,
                )
                self.logger.log_exception(KeyboardInterrupt)
                self.logger.log_info(
                    "Default Weights Saved  at KeyBoard Interruption at epoch : {}".format(
                        ongoing_epoch
                    )
                )
                self.logger.log_info("Total Training Time : {}".format(str(total_time)))

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
                self.logger.log_info("Total Training Time : {}".format(str(total_time)))
                print("\n")

                print_format.colored_dual_string_print_with_brackets(
                    "Run Time", total_time, "green", "yellow", attrs=["bold", "blink"]
                )
                sys.exit(str(traceback.format_exc()))

        print_format.colored_dual_string_print_with_brackets(
            "DONE", "Training", "green", "yellow", attrs=["bold", "blink"]
        )
        data_viz.save_viz()
        data_viz.terminate_viz()
        end_time = time.time()
        total_time = date_time_utility.get_time(end_time - start_time)
        self.logger.log_info("Total Training Time : {}".format(str(total_time)))

    def initialize_required_training_param(self):
        optimizer = self.parameter_loader.load_optimizer(self.parameter_loader.model)
        loss = self.parameter_loader.load_loss()
        lr_scheduler = self.parameter_loader.load_lr_scheduler(optimizer)
        model, optimizer, learning_rate, starting_epoch, step = self.parameter_loader.load_model_weights_and_params(
            optimizer, status=self.status, weight_path=self.default_weights_path
        )
        return model, optimizer, loss, lr_scheduler, learning_rate, starting_epoch, step

    def create_current_run_config_helper(self):
        current_run_config = os.path.join(
            os.path.join(self.training_path, self.version), "configuration.yaml"
        )

        yaml_file_object = open(current_run_config, "w")
        data = {
            "Date": str(date_time_utility.get_date()),
            "Version": self.version,
            "Status": self.status,
        }
        parameter_data = {**data, **self.training_configuration}

        yaml_utility.write_config_yaml(parameter_data, yaml_file_object)

    def initiate_data_viz(self):
        data_viz = DataViz(
            env_name=str(
                self.training_configuration["experiment_name"]
                + self.training_configuration["model"]
                + self.version
            ),
            logger=self.logger,
        )
        return data_viz
