import fire
import os

from setup.train_config import TrainConfig
from ml.pt.loader import Loader
from ml.trainer import Trainer
from ml.pt.factory import Plugin
from utils.logger import create_logger


def main(config):
    plugin = Plugin(config)
    plugin.load_trainer_plugin()
    loader = Loader(config)
    trainer = Trainer(plugin, loader, config)
    trainer.training()


def default(config_path, parameter_path):
    config = TrainConfig.create_config(config_path, parameter_path)
    config.set_property("training_state", "DEFAULT")
    config.set_additional_configuration()

    save_path = os.path.join(config.training_path, config.version)

    config.write_config(save_path)
    create_logger(config.root_folder, config.experiment_name)
    main(config)


def resume(resume_config_path, param_path=None):
    config = TrainConfig.create_config(resume_config_path, param_path)
    config.set_run_property(os.path.dirname(resume_config_path))
    config.update_property("training_state", "RESUME")

    save_path = os.path.join(config.training_path, config.version)

    config.write_config(save_path)
    create_logger(config.root_folder, config.experiment_name)
    main(config)


if __name__ == "__main__":
    fire.Fire()
