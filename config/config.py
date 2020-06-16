import os

import yaml

from pyjavaproperties import Properties


class PtConfig(object):
    def __init__(self, config_path, restriction, plugin):
        self._config = self.read_config_yaml(config_path)
        self._config_section = restriction
        self._run_config = self.simplify_run_config()
        self._additional_config = Properties()
        self._plugin = plugin

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
    def additional_config(self):
        return self._additional_config

    @additional_config.setter
    def additional_config(self, value: Properties):
        self._additional_config = value

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
        if property_name in self.additional_config:
            return self.additional_config[property_name]
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
    def read_properties_file(config_path):
        prop = Properties()
        prop.load(open(PtConfig.get_properties_file(config_path)))
        return prop

    @staticmethod
    def get_properties_file(config_path):
        properties_file = os.path.join(config_path, "chronos.properties")
        return properties_file

    def write_config(self, save_path):
        yaml_file_object = open(os.path.join(save_path, "configuration.yaml"), "w")
        yaml.dump(self.get_run_config(), yaml_file_object, default_flow_style=False)
        self.additional_config.store(open(self.get_properties_file(save_path), "w"))
