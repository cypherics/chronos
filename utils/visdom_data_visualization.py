import subprocess

import time
import visdom

import torch
import numpy as np


class DataViz:
    def __init__(self, env_name, logger, flush_existing_env=False):
        self.env_name = env_name
        self.port = 8080
        self.flush_existing_env = flush_existing_env
        self.logger = logger

        self.image_plot = "test_images_plot"
        self.epoch_loss_plot = "epoch_loss_plot"
        self.learning_rate_plot = "lr_plot"

        self.viz_server, self.viz_process = self.create_viz_server()
        self.create_viz_graph_object()

    def create_viz_server(self):
        viz_process = subprocess.Popen(
            [
                "python3",
                "-m",
                "visdom.server",
                "-port={}".format(self.port),
                "-logging_level={}".format("ERROR"),
            ],
            shell=False,
        )

        time.sleep(2)

        viz_server = visdom.Visdom(port=self.port, env=self.env_name)

        if self.flush_existing_env:
            viz_server.delete_env(viz_server.env)

        return viz_server, viz_process

    def create_line_plot(self, xlabel, ylabel, title, win):
        if not self.viz_server.win_exists(win, env=self.env_name):
            self.viz_server.line(
                X=torch.zeros((1)).cpu(),
                Y=torch.zeros((1)).cpu(),
                opts=dict(xlabel=xlabel, ylabel=ylabel, title=title),
                win=win,
            )

    def create_image_plot(self, img_shape, title, win):
        if not self.viz_server.win_exists(win, env=self.env_name):
            return self.viz_server.images(
                np.random.rand(*img_shape), opts=dict(title=title), win=win
            )

    def create_viz_graph_object(self):
        self.create_line_plot(
            xlabel="Epoch",
            ylabel="Learning Rate",
            title="Learning Rate Graph",
            win=self.learning_rate_plot,
        )
        self.create_line_plot(
            xlabel="Epoch",
            ylabel="Loss",
            title="Epoch Loss Graph",
            win=self.epoch_loss_plot,
        )
        self.create_image_plot((3, 256, 512), title="Test Images", win=self.image_plot)

    def plot_learning_rate(self, lr, epoch):
        # Plotting Learning Rate Graph
        self.viz_server.line(
            X=torch.ones(1).cpu() * epoch,
            Y=torch.Tensor([lr]).cpu(),
            win=self.learning_rate_plot,
            name="learning rate",
            update="append",
        )
        self.save_viz()

    def plot_train_loss(self, loss, epoch):
        # Plotting Epoch Graph
        self.viz_server.line(
            X=torch.ones(1).cpu() * epoch,
            Y=torch.Tensor([loss.item()]).cpu(),
            win=self.epoch_loss_plot,
            name="train",
            update="append",
        )
        self.save_viz()

    def plot_val_loss(self, loss, epoch):
        # Plotting Epoch Graph
        self.viz_server.line(
            X=torch.ones(1).cpu() * epoch,
            Y=torch.Tensor([loss]).cpu(),
            win=self.epoch_loss_plot,
            name="validation",
            update="append",
        )
        self.save_viz()

    def plot_test_image(self, predicted_images):
        self.viz_server.image(
            torch.FloatTensor(predicted_images.swapaxes(0, -1).swapaxes(-1, 1)),
            win=self.image_plot,
            opts=dict(title="Test Images"),
        )
        self.save_viz()

    def save_viz(self):
        self.viz_server.save([self.env_name])

    def terminate_viz(self):
        self.viz_process.terminate()
