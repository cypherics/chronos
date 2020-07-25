import os


def make_directory(current_dir, folder_name):
    new_dir = os.path.join(current_dir, folder_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def create_state_path(pth, version):
    version_path = make_directory(pth, version)
    state_path = make_directory(version_path, "state")
    default_weights_path = os.path.join(state_path, "default_state.pt")
    best_weights_path = os.path.join(state_path, "best_state.pt")
    return default_weights_path, best_weights_path


def create_chk_path(pth, exp_name, model, version):
    version_path = make_directory(pth, version)
    chk_path = make_directory(version_path, "chk_pt")
    weights_path = os.path.join(
        chk_path, "{}_{}_{}_chk.pt".format(exp_name, model, version)
    )
    return weights_path


def level_2_folder_creation(root, level_1, level_2):
    root_folder = make_directory(os.getcwd(), "exp_zoo/" + root)
    level_1_folder_path = make_directory(root_folder, level_1)
    level_2_folder_path = make_directory(level_1_folder_path, level_2)

    return root_folder, level_1_folder_path, level_2_folder_path


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
