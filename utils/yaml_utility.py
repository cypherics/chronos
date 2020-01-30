import yaml

from utils.directory_handler import get_resume_version_config


def read_config_yaml(file_path: str) -> dict:
    with open(file_path, "r") as reader:
        config = yaml.load(reader)
    return config


def write_config_yaml(data, yaml_file_object):
    yaml.dump(data, yaml_file_object, default_flow_style=False)


def extract_training_config(config, parameter):
    training_state = config["training_state"]
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

    if training_state == "RESUME":
        resume_config_path = get_resume_version_config(
            training_configurations["experiment_name"], training_configurations["model"]
        )
        resume_config = read_config_yaml(resume_config_path)
        resume_config["training_state"] = training_state
        return resume_config
    else:
        return training_configurations
