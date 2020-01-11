import yaml

from utils.directory_handler import get_resume_version_config


def read_config_yaml(file_path: str) -> dict:
    with open(file_path, "r") as reader:
        config = yaml.load(reader)
    return config


def write_config_yaml(data, yaml_file_object):
    yaml.dump(data, yaml_file_object, default_flow_style=False)


def current_run_config(config, parameter_config):
    training_configurations = dict()
    training_state = parameter_config["training_state"]
    if training_state == "RESUME":
        resume_config_path = get_resume_version_config(
            config["experiment_name"], config["model"]
        )
        resume_config = read_config_yaml(resume_config_path)
        resume_config["training_state"] = training_state
        training_configurations = resume_config
    else:
        for key, value in config.items():
            training_configurations[key] = value
            if key in parameter_config:
                if value is None:
                    training_configurations[value] = None
                else:
                    training_configurations[value] = parameter_config[key][value]

        for key, value in parameter_config.items():
            if key not in training_configurations:
                training_configurations[key] = value
    return training_configurations
