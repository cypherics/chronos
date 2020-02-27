import random
import time
import os
import numpy as np

import torch
import tqdm

from ml.commons.data_visualization import DataViz
from ml.commons.utils import torch_tensor_conversion
from utils import directory_handler, path_creation, date_time_utility, yaml_utility
from utils.logger import LogDecorator


class Trainer:
    def __init__(self, plugin, loader, training_configuration):
        self.training_configuration = training_configuration
        self.loader = loader
        self.plugin = plugin
        self.train_state, self.train_version = self.get_run_version_state()

        self.default_weights_path, self.best_weights_path = path_creation.create_weights_path(
            self.training_configuration["training_path"], self.train_version
        )

    @LogDecorator()
    def get_run_version_state(self):
        status = self.training_configuration["training_state"]
        current_version, previous_version = directory_handler.make_version(
            self.training_configuration["training_path"]
        )
        if status == "RESUME":
            current_version = self.training_configuration["Version"]
        assert status in ["TRANSFER_LEARNING", "RESUME", "DEFAULT"]
        print(" STATE : {}, VERSION: {}".format(status, current_version))
        return status, current_version

    @LogDecorator()
    def init_training_param(self):
        optimizer = self.loader.load_optimizer(self.plugin.model)
        loss = self.plugin.loss
        lr_scheduler = self.loader.load_lr_scheduler(optimizer)
        model, optimizer, learning_rate, starting_epoch, step = self.loader.load_training_state(
            self.plugin.model, optimizer, status=self.train_state, weight_path=self.default_weights_path
        )
        return model, optimizer, loss, lr_scheduler, learning_rate, starting_epoch, step

    @LogDecorator()
    def training(self):
        start_time = time.time()
        model, optimizer, loss, lr_scheduler, learning_rate, starting_epoch, step = (
            self.init_training_param()
        )
        train_loader, val_loader, test_loader = self.loader.load_loader()
        data_viz = self.initiate_data_viz()
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
                    calculated_loss = loss(outputs=outputs, **input_data)
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
                    data_viz.plot_iteration_loss(mean_loss, step)
                    step += 1
                    if i and i % 20 == 0:
                        predicted_images = self.plugin.validation.inference(
                            model,
                            i,
                            ongoing_epoch,
                            test_loader,
                            save_path=os.path.join(self.training_configuration["training_path"],
                                                   self.train_version),
                        )
                        data_viz.plot_test_image(predicted_images)
                #
                data_viz.plot_learning_rate(lr, ongoing_epoch)
                #
                data_viz.plot_train_loss(mean_loss, ongoing_epoch)

                tq.close()

                valid_metrics = self.plugin.validation.perform_validation(
                    model=model,
                    loss_function=loss,
                    valid_loader=val_loader,
                )
                valid_loss = valid_metrics["valid_loss"]
                valid_losses.append(valid_loss)

                if lr_scheduler is not None:
                    lr_scheduler.step(epoch=ongoing_epoch + 1, valid_loss=valid_loss)

                print(*valid_metrics.items(), sep='\n')

                # self.early_stopping.step(valid_loss)
                print("DEFAULT CHECKPOINT SAVED / {}".format(ongoing_epoch))

                data_viz.plot_val_loss(valid_loss, ongoing_epoch)

                if (previous_min_loss is None) or (valid_loss < previous_min_loss):
                    previous_min_loss = valid_loss

                    self.save_check_point(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        epoch=ongoing_epoch,
                        lr=lr,
                        model_path=self.best_weights_path,
                    )
                    print("BEST CHECKPOINT SAVED / {}".format(ongoing_epoch))

                self.save_check_point(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=ongoing_epoch,
                    lr=lr,
                    model_path=self.default_weights_path,
                )
                print("DEFAULT CHECKPOINT SAVED / {}".format(ongoing_epoch))

            except KeyboardInterrupt:
                tq.close()
                data_viz.save_viz()
                data_viz.terminate_viz()

                end_time = time.time()
                total_time = date_time_utility.get_time(end_time - start_time)
                print("Run Time : {}".format(total_time))
                print("CHECKPOINT SAVED / {}".format(ongoing_epoch))
                self.save_check_point(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=ongoing_epoch,
                    lr=lr,
                    model_path=self.default_weights_path,
                )

                raise KeyboardInterrupt

            except Exception as ex:
                data_viz.save_viz()
                data_viz.terminate_viz()
                end_time = time.time()
                print("Exception")
                print("Run Time : {}".format(end_time))
                raise ex

        data_viz.save_viz()
        data_viz.terminate_viz()
        end_time = time.time()
        print("Run Time : {}".format(end_time))
        print("Training Complete")

    @LogDecorator()
    def initiate_data_viz(self):
        data_viz = DataViz(
            env_name=str(
                self.training_configuration["experiment_name"]
                + str("_")
                + self.training_configuration["model"]
                + str("_")
                + self.train_version
            ),
        )
        return data_viz

    @staticmethod
    def save_check_point(model, optimizer, epoch, step, lr, model_path):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "starting_epoch": epoch,
                "step": step,
                "lr": lr,
            },
            str(model_path),
        )

    def create_current_run_config_helper(self):
        current_run_config = os.path.join(
            os.path.join(self.training_configuration["training_path"], self.train_version), "configuration.yaml"
        )

        yaml_file_object = open(current_run_config, "w")
        data = {
            "Date": str(date_time_utility.get_date()),
            "Version": self.train_version,
            "Status": self.train_state,
        }
        parameter_data = {**data, **self.training_configuration}

        yaml_utility.write_config_yaml(parameter_data, yaml_file_object)
