import os

from ml.pt.factory import Plugin
from ml.pt.loader import Loader
from ml.trainer import Trainer
from setup.train_config import TrainConfig
from utils.logger import create_logger

CONFIG_RESTRICTION = ["DATASET", "MODEL", "TRAIN"]


class Default:
    def __init__(self, config_path):
        self.config_path = config_path

    def run(self):
        config = TrainConfig.create_config(self.config_path, CONFIG_RESTRICTION)
        config.set_property("training_state", "DEFAULT")
        config.set_additional_property()

        save_path = os.path.join(config.training_path, config.version)

        config.write_config(save_path)
        create_logger(config.root_folder, config.experiment_name)
        plugin = Plugin(config)
        plugin.load_trainer_plugin()
        loader = Loader(config)
        trainer = Trainer(plugin, loader, config)
        trainer.training()
