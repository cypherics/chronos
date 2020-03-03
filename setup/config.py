import os

import yaml

from utils import directory_handler, date_time_utility


class Config(object):
    def __init__(self, conf):
        self._config = conf

    def get_property(self, property_name):
        if property_name not in self._config.keys():
            raise KeyError
        return self._config[property_name]

    def update_property(self, property_name, property_value):
        if property_name not in self._config.keys():
            raise KeyError
        self._config[property_name] = property_value

    def set_property(self, property_name, property_value):
        if property_name not in self._config.keys():
            self._config[property_name] = property_value

    def get_sub_property(self, head_property, property_name):
        if head_property not in self._config.keys():
            raise KeyError
        return self.recursive_dict_fn(self._config, property_name)

    def recursive_dict_fn(self, recursive_dict, property_name, dict_value=None):

        for key, value in recursive_dict.items():
            if isinstance(value, dict):
                dict_value = self.recursive_dict_fn(value, property_name, dict_value)
            else:
                if key == property_name:
                    return value
        return dict_value

    @staticmethod
    def folder_creation(exp_name_folder, model_name, folder_type="training"):
        root_folder = directory_handler.make_directory(
            os.getcwd(), "run/" + exp_name_folder
        )
        model_folder_path = directory_handler.make_directory(root_folder, model_name)
        folder_type_path = directory_handler.make_directory(model_folder_path, folder_type)

        return root_folder, model_folder_path, folder_type_path

    @staticmethod
    def read_config_yaml(file_path: str) -> dict:
        with open(file_path, "r") as reader:
            config = yaml.load(reader)
        return config

    @staticmethod
    def merge_config(config, parameter):
        train_config = dict()
        parameter_config = dict()
        for sec in config:
            train_config[sec] = config[sec]

        for sec in parameter:
            if parameter[sec] is None:
                parameter_config[sec] = None
            else:
                for key, value in parameter[sec].items():
                    parameter_config[sec] = key
                    parameter_config[key] = value

        training_configurations = {**train_config, **parameter_config}
        return training_configurations

    @classmethod
    def create_config(cls, config_path, parameter_path=None):
        conf = cls.read_config_yaml(config_path) if parameter_path is None else \
            cls.merge_config(cls.read_config_yaml(config_path), cls.read_config_yaml(parameter_path))

        return cls(conf)

    def write_config(self, save_path):
        yaml_file_object = open(save_path, "w")
        data = {
            "Date": str(date_time_utility.get_date()),
        }
        parameter_data = {**data, **self._config}
        yaml.dump(parameter_data, yaml_file_object, default_flow_style=False)

    @staticmethod
    def create_version(directory):
        subdir = os.listdir(directory)
        if len(subdir) == 0:
            version_number = 1
        else:
            existing_version = list()
            for sub in subdir:
                version_number = sub[1:]
                existing_version.append(int(version_number))
            existing_version.sort()
            version_number = existing_version[-1] + 1
        current_version = "v"+str(version_number)

        return current_version

