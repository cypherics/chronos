import os
from utils.directory_utils import directory_handler


def create_necessary_folder(exp_name_folder, model_name):
    root_folder = directory_handler.make_directory(
        os.getcwd(), "checkpoints/" + exp_name_folder
    )
    model_folder_path = directory_handler.make_directory(root_folder, model_name)
    training_path = directory_handler.make_directory(model_folder_path, "training")

    return root_folder, model_folder_path, training_path


def create_weights_path(training_path, version):
    version_path = directory_handler.make_directory(training_path, version)
    default_weights_path = os.path.join(version_path, "default_checkpoint.pt")
    best_weights_path = os.path.join(version_path, "best_checkpoint.pt")
    return default_weights_path, best_weights_path
