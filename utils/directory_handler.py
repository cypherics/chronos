import os


def make_directory(current_dir, folder_name):
    new_dir = os.path.join(current_dir, folder_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def make_version(directory):
    subdir = os.listdir(directory)
    if len(subdir) == 0:
        current_version = "v1"
        previous_version = "v0"
    else:
        current_version = "v" + str(len(subdir) + 1)
        previous_version = "v" + str(len(subdir) + 1 - 1)

    return current_version, previous_version


def get_resume_version_config(exp_name, model):
    resume_path = os.path.join(
        *[os.getcwd(), "checkpoints", exp_name, model, "training"]
    )
    current_version, previous_version = make_version(resume_path)

    return os.path.join(*[resume_path, previous_version, "configuration.yaml"])
