from ml.loader.data_loader import DominusDataLoader
from ml.loader.model_loader import ModelLoader
from ml.loader.validation_loader import DominusValidation
from ml.trainer import Trainer
from utils import yaml_utility, print_format
from utils.directory_utils import path_creation
from utils.logger import Logger

CONFIG = "ml/configuration/training_config.yaml"
CONFIG_PARAMETER_ASSIGN = "ml/configuration/training_parameter_assign.yaml"


def main():
    config = yaml_utility.read_config_yaml(CONFIG)
    config_parameter_assign = yaml_utility.read_config_yaml(CONFIG_PARAMETER_ASSIGN)

    training_configurations = yaml_utility.current_run_config(
        config, config_parameter_assign
    )

    root_folder, model_folder_path, training_path = path_creation.create_necessary_folder(
        training_configurations["experiment_name"], training_configurations["model"]
    )

    log = Logger(root_folder, training_configurations["experiment_name"])

    print_format.colored_single_string_print_with_brackets("Loading", "red")
    print_format.print_tab_fancy("Inference and DataLoader", "yellow")
    data_loader = DominusDataLoader(training_configurations, log)
    validation_loader = DominusValidation(training_configurations, log)
    print_format.print_tab_fancy("Model", "yellow")
    model_loader = ModelLoader(training_configurations, log)

    trainer = Trainer(
        data_loader,
        validation_loader,
        model_loader,
        training_configurations,
        log,
        training_path,
    )
    trainer.create_current_run_config_helper()
    trainer.training()


main()
