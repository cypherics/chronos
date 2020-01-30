import os
import sys
import traceback

import fire

from ml.pt_model import ModelPt
from ml.pt_loader import LoaderPt
from ml.pt_lr_finder import LrFinderPt
from utils import yaml_utility, print_format, directory_handler
from utils.logger import Logger


def main(config, parameter):
    config = yaml_utility.read_config_yaml(config)
    config_parameter_assign = yaml_utility.read_config_yaml(parameter)

    lr_finder_configurations = yaml_utility.extract_training_config(
        config, config_parameter_assign
    )
    root_folder = directory_handler.make_directory(
        os.getcwd(), "checkpoints/" + "LR_Finder"
    )

    log = Logger(root_folder, lr_finder_configurations["experiment_name"])

    try:
        assert lr_finder_configurations["training_state"] == "DEFAULT"
        assert (
            "start_lr"
            and "end_lr" in lr_finder_configurations["initial_assignment"].keys()
        )
    except AssertionError as ex:
        print_format.colored_dual_string_print_with_brackets(
            str("ERROR"),
            "LR Finder parameter error, check log",
            "red",
            "cyan",
            attrs=["blink"],
        )
        log.log_exception(ex)
        sys.exit(str(traceback.format_exc()))

    print_format.colored_single_string_print_with_brackets("Loading", "red")
    print_format.print_tab_fancy("Inference / DataLoader ", "yellow")
    loader_pt = LoaderPt(lr_finder_configurations, log)
    print_format.print_tab_fancy("Model", "yellow")
    model_pt = ModelPt(lr_finder_configurations, log)

    lr_finder_pt = LrFinderPt(loader_pt, model_pt, lr_finder_configurations, log)
    lr_finder_pt.find_lr()


if __name__ == "__main__":
    fire.Fire(main)
