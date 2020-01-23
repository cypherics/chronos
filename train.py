import fire

from ml.pt_model import ModelPt
from ml.pt_loader import LoaderPt
from ml.pt_trainer import TrainerPt
from utils import yaml_utility, print_format, path_creation
from utils.logger import Logger


def main(config, parameter):
    config = yaml_utility.read_config_yaml(config)
    config_parameter_assign = yaml_utility.read_config_yaml(parameter)

    training_configurations = yaml_utility.extract_training_config(
        config, config_parameter_assign
    )

    root_folder, model_folder_path, training_path = path_creation.create_necessary_folder(
        training_configurations["experiment_name"], training_configurations["model"]
    )

    log = Logger(root_folder, training_configurations["experiment_name"])

    print_format.colored_single_string_print_with_brackets("Loading", "red")
    print_format.print_tab_fancy("Inference / DataLoader ", "yellow")
    loader_pt = LoaderPt(training_configurations, log)
    print_format.print_tab_fancy("Model", "yellow")
    model_pt = ModelPt(training_configurations, log)

    trainer_pt = TrainerPt(
        loader_pt, model_pt, training_configurations, log, training_path
    )
    trainer_pt.create_current_run_config_helper()
    trainer_pt.training()


if __name__ == '__main__':
    fire.Fire(main)
