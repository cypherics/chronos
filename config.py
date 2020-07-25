import os

import yaml

from pyjavaproperties import Properties

from utils.directory_ops import (
    level_2_folder_creation,
    create_version,
    create_state_path,
    create_chk_path,
)


class Config(object):
    def __init__(self, config_path, restriction, plugin_name):
        self._config = self.read_config_yaml(config_path)
        self._config_section = restriction
        self._run_config = self.simplify_run_config()
        self._plugin = plugin_name
        self._resume = self.resume_config(os.path.dirname(config_path))
        self._additional_property = (
            self.read_properties_file(os.path.dirname(config_path))
            if self.resume
            else Properties()
        )

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
    def model_name(self):
        return self.get_property("MODEL_NAME")

    @property
    def model_param(self):
        param = self.get_property("MODEL_PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def loss_name(self):
        return self.get_sub_property("LOSS", "NAME")

    @property
    def loss_param(self):
        param = self.get_sub_property("LOSS", "PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def scheduler_name(self):
        return self.get_sub_property("SCHEDULER", "NAME")

    @property
    def scheduler_param(self):
        param = self.get_sub_property("SCHEDULER", "PARAM")
        return param if param is not None else {"NA": "NA"}

    @property
    def optimizer_name(self):
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

    @property
    def resume(self):
        return self._resume

    @property
    def plugin(self):
        return self._plugin

    @property
    def config_restriction(self):
        return self._config_section

    @config_restriction.setter
    def config_restriction(self, value):
        self._config_section = value

    @property
    def additional_property(self):
        return self._additional_property

    @additional_property.setter
    def additional_property(self, value: Properties):
        self._additional_property = value

    def simplify_run_config(self):
        conf = dict()
        run_config = self.get_run_config()
        for sec in run_config:
            for key, value in run_config[sec].items():
                conf[key] = value
        return conf

    def get_run_config(self):
        conf = dict()
        restricted_section = self.config_restriction
        if restricted_section is not None:
            for sec in self._config:
                if sec in restricted_section:
                    conf[sec] = self._config[sec]
        else:
            for sec in self._config:
                conf[sec] = self._config[sec]
        return conf

    def get_property(self, property_name):
        if property_name in self._run_config.keys():
            return self._run_config[property_name]
        else:
            raise KeyError

    def get_additional_property(self, property_name):
        if property_name in self.additional_property:
            return self.additional_property[property_name]
        else:
            raise KeyError

    def get_sub_property(self, head_property, property_name):
        if head_property not in self._run_config.keys():
            raise KeyError
        return self.recursive_dict_fn(self._run_config[head_property], property_name)

    def recursive_dict_fn(self, recursive_dict, property_name, dict_value=None):

        for key, value in recursive_dict.items():
            if key == property_name:
                return value
            elif isinstance(value, dict):
                dict_value = self.recursive_dict_fn(value, property_name, dict_value)

        return dict_value

    @staticmethod
    def read_config_yaml(file_path: str) -> dict:
        with open(file_path, "r") as reader:
            config = yaml.load(reader)
        return config

    @staticmethod
    def read_properties_file(config_dir):
        prop = Properties()
        prop.load(open(Config.get_properties_file(config_dir)))
        return prop

    @staticmethod
    def get_properties_file(config_dir):
        properties_file = os.path.join(config_dir, "chronos.properties")
        return properties_file

    def write_config(self, save_path):
        yaml_file_object = open(os.path.join(save_path, "configuration.yaml"), "w")
        yaml.dump(self.get_run_config(), yaml_file_object, default_flow_style=False)
        self.additional_property.store(open(self.get_properties_file(save_path), "w"))

    def resume_config(self, config_dir):
        return os.path.exists(self.get_properties_file(config_dir))

    def generate_additional_train_property(self):
        additional_property = Properties()
        additional_property["root_folder"], _, additional_property[
            "training_path"
        ] = level_2_folder_creation(self.experiment_name, "train", self.model_name)
        additional_property["Version"] = create_version(
            additional_property["training_path"]
        )

        additional_property["default_state"], additional_property[
            "best_state"
        ] = create_state_path(
            additional_property["training_path"], additional_property["Version"]
        )

        additional_property["chk_pth"] = create_chk_path(
            additional_property["training_path"],
            self.experiment_name,
            self.model_name,
            additional_property["Version"],
        )

        self.additional_property = additional_property
