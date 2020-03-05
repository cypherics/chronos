import os
from utils import directory_handler

from setup.config import Config


class TrainConfig(Config):
    def __init__(self, conf, config_restriction):
        super().__init__(conf, config_restriction)

    def create_weights_path(self):
        version_path = directory_handler.make_directory(
            self.training_path, self.version
        )
        default_weights_path = os.path.join(version_path, "default_checkpoint.pt")
        best_weights_path = os.path.join(version_path, "best_checkpoint.pt")
        return default_weights_path, best_weights_path

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
    def problem_type(self):
        return self.get_property("ML_TYPE")

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
        return self.get_property("MODEL_PARAM")

    @property
    def loss(self):
        return self.get_sub_property("LOSS", "NAME")

    @property
    def loss_param(self):
        return self.get_sub_property("LOSS", "PARAM")

    @property
    def scheduler(self):
        return self.get_sub_property("SCHEDULER", "NAME")

    @property
    def scheduler_param(self):
        return self.get_sub_property("SCHEDULER", "PARAM")

    @property
    def optimizer(self):
        return self.get_sub_property("OPTIMIZER", "NAME")

    @property
    def optimizer_param(self):
        return self.get_sub_property("OPTIMIZER", "PARAM")

    @property
    def training_state(self):
        return self.get_property("training_state")

    @property
    def training_path(self):
        return self.get_property("training_path")

    @training_path.setter
    def training_path(self, value):
        self.set_property("training_path", value)

    @property
    def root_folder(self):
        return self.get_property("root_folder")

    @root_folder.setter
    def root_folder(self, value):
        self.set_property("root_folder", value)

    @property
    def version(self):
        return self.get_property("Version")

    @version.setter
    def version(self, value):
        self.set_property("Version", value)

    @property
    def chk_path(self):
        return self.get_property("chk_path")

    @chk_path.setter
    def chk_path(self, value):
        self.set_property("chk_path", value)

    @property
    def best_chk_path(self):
        return self.get_property("best_chk_path")

    @best_chk_path.setter
    def best_chk_path(self, value):
        self.set_property("best_chk_path", value)

    def set_additional_property(self):
        self.root_folder, _, self.training_path = self.folder_creation(
            self.experiment_name, self.model
        )
        self.version = (
            self.create_version(self.training_path)
            if self.training_state == "DEFAULT"
            else self.get_property("Version")
        )
        self.chk_path, self.best_chk_path = self.create_weights_path()
