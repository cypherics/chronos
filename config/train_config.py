import os

from pyjavaproperties import Properties


from config.config import PtConfig
from utils.directory_handler import (
    create_version,
    level_2_folder_creation,
    create_state_path,
    create_chk_path,
)


class TrainConfig(PtConfig):
    def __init__(self, config_path, restriction, plugin):
        super().__init__(config_path, restriction, plugin)
        self.additional_config = self.generate_additional_config(
            os.path.dirname(config_path)
        )

    def generate_additional_config(self, config_path):
        if os.path.exists(self.get_properties_file(config_path)):
            return self.read_properties_file(config_path)
        else:
            additional_config = Properties()
            additional_config["root_folder"], _, additional_config[
                "training_path"
            ] = level_2_folder_creation(self.experiment_name, "train", self.model)
            additional_config["Version"] = create_version(
                additional_config["training_path"]
            )

            additional_config["default_state"], additional_config[
                "best_state"
            ] = create_state_path(
                additional_config["training_path"], additional_config["Version"]
            )

            additional_config["chk_pth"] = create_chk_path(
                additional_config["training_path"],
                self.experiment_name,
                self.model,
                additional_config["Version"],
            )

            return additional_config

    @property
    def transformation(self):
        return self.get_property("TRANSFORMATION")

    @property
    def root(self):
        return self.get_property("ROOT")

    @property
    def experiment_name(self):
        return self.get_property("EXP_NAME")

    @property
    def normalization(self):
        return self.get_property("NORMALIZATION")

    @property
    def batch_size(self):
        return self.get_property("BATCH")

    @property
    def n_epochs(self):
        return self.get_property("EPOCH")

    @property
    def model_input_dimension(self):
        return self.get_property("IMAGE_DIM")

    @property
    def model(self):
        return self.get_property("MODEL_NAME")

    @property
    def model_param(self):
        param = self.get_property("MODEL_PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def loss(self):
        return self.get_sub_property("LOSS", "NAME")

    @property
    def loss_param(self):
        param = self.get_sub_property("LOSS", "PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def scheduler(self):
        return self.get_sub_property("SCHEDULER", "NAME")

    @property
    def scheduler_param(self):
        param = self.get_sub_property("SCHEDULER", "PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def optimizer(self):
        return self.get_sub_property("OPTIMIZER", "NAME")

    @property
    def optimizer_param(self):
        param = self.get_sub_property("OPTIMIZER", "PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def training_path(self):
        return self.get_additional_property("training_path")

    @property
    def root_folder(self):
        return self.get_additional_property("root_folder")

    @property
    def version(self):
        return self.get_additional_property("Version")

    @property
    def default_state(self):
        return self.get_additional_property("default_state")

    @property
    def best_state(self):
        return self.get_additional_property("best_state")

    @property
    def chk_pth(self):
        return self.get_additional_property("chk_pth")
