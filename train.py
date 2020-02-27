import fire

from ml.pt.loader import Loader
from ml.pt.trainer import Trainer
from utils import yaml_utility, path_creation
from utils.logger import create_logger
from ml.ml_plugin import PluginCollection
from ml.pt.plugin_pt import PluginPt


def main(training_configurations):

    root_folder, model_folder_path, training_path = path_creation.create_necessary_folder(
        training_configurations["experiment_name"], training_configurations["model"]
    )

    training_configurations["training_path"] = training_path
    create_logger(root_folder, training_configurations["experiment_name"])

    plugin_pt = PluginPt(PluginCollection(training_configurations["problem_type"], training_configurations))
    loader = Loader(plugin_pt, training_configurations)
    trainer = Trainer(plugin_pt, loader, training_configurations)
    trainer.create_current_run_config_helper()
    trainer.training()


def default(config, parameter):
    config = yaml_utility.read_config_yaml(config)
    config_parameter_assign = yaml_utility.read_config_yaml(parameter)

    training_configurations = yaml_utility.extract_training_config(
        config, config_parameter_assign
    )
    training_configurations["training_state"] = "DEFAULT"
    main(training_configurations)


def resume(resume_configurations):
    resume_configurations = yaml_utility.read_config_yaml(resume_configurations)
    resume_configurations["training_state"] = "RESUME"
    main(resume_configurations)


def transfer_learning(config, parameter, weights_file):
    config = yaml_utility.read_config_yaml(config)
    config_parameter_assign = yaml_utility.read_config_yaml(parameter)

    training_configurations = yaml_utility.extract_training_config(
        config, config_parameter_assign
    )
    training_configurations["training_state"] = "TRANSFER_LEARNING"
    training_configurations["transfer_weights_path"] = weights_file
    main(training_configurations)


if __name__ == "__main__":
    fire.Fire()
