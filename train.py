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


def default(config, parameter):
    config = TrainConfig.create_config(config, parameter)
    config.set_property("training_state", "DEFAULT")
    config.set_additional_configuration()

    save_path = os.path.join(
        os.path.join(
            config.training_path, config.version
        ),
        "configuration.yaml",
    )
    config.write_config(save_path)
    create_logger(config.root_folder, config.experiment_name)
    main(config)


def resume(resume_configurations):
    config = TrainConfig.create_config(resume_configurations)
    config.update_property("training_state", "RESUME")
    config.set_additional_configuration()
    save_path = os.path.join(
        os.path.join(
            config.training_path, config.version
        ),
        "configuration.yaml",
    )
    config.write_config(save_path)
    create_logger(config.root_folder, config.experiment_name)
    main(config)


if __name__ == "__main__":
    fire.Fire()
