import os

import yaml

from utils import directory_handler
from pyjavaproperties import Properties


class Config(object):
    def __init__(self, conf, config_restriction):
        self._config = conf
        self.config_restriction = config_restriction
        self._run_config = self.simplify_run_config()
        self._run_property = Properties()

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
        if property_name in self._run_property:
            return self._run_property[property_name]
        elif property_name in self._run_config.keys():
            return self._run_config[property_name]
        else:
            raise KeyError

    def update_property(self, property_name, property_value):
        if property_name not in self._run_property:
            raise KeyError
        self._run_property[property_name] = property_value

    def set_property(self, property_name, property_value):
        if property_name not in self._run_config.keys():
            self._run_property[property_name] = property_value

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
    def folder_creation(exp_name_folder, model_name, folder_type="training"):
        root_folder = directory_handler.make_directory(
            os.getcwd(), "run/" + exp_name_folder
        )
        model_folder_path = directory_handler.make_directory(root_folder, model_name)
        folder_type_path = directory_handler.make_directory(
            model_folder_path, folder_type
        )

        return root_folder, model_folder_path, folder_type_path

    @staticmethod
    def read_config_yaml(file_path: str) -> dict:
        with open(file_path, "r") as reader:
            config = yaml.load(reader)
        return config

    @staticmethod
    def read_properties_file(config_path):
        prop = Properties()
        prop.load(open(Config.get_properties_file(config_path)))
        return prop

    @staticmethod
    def get_properties_file(config_path):
        properties_file = os.path.join(config_path, "dominus.properties")
        return properties_file

    def write_config(self, save_path):
        yaml_file_object = open(os.path.join(save_path, "configuration.yaml"), "w")
        yaml.dump(self.get_run_config(), yaml_file_object, default_flow_style=False)
        self._run_property.store(open(self.get_properties_file(save_path), "w"))

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
        current_version = "v" + str(version_number)

        return current_version

    @classmethod
    def create_config(cls, config_path, config_restriction=None):
        conf = cls.read_config_yaml(config_path)
        return cls(conf, config_restriction)

    def set_run_property(self, path):
        self._run_property = self.read_properties_file(path)

    @property
    def config_restriction(self):
        return self._config_section

    @config_restriction.setter
    def config_restriction(self, value):
        self._config_section = value
