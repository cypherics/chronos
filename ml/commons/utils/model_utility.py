import torch

from ml.pt.logger import PtLogger


@PtLogger(debug=True)
def get_gpu_device_ids():
    device_id = list()
    separator = ","
    gpu_device_available = torch.cuda.device_count()
    for i in range(gpu_device_available):
        device_id.append(str(i))
    device_id = separator.join(device_id)
    return device_id


@PtLogger()
def get_current_state(weight_path):
    state = torch.load(str(weight_path), map_location="cpu")
    return state


@PtLogger()
def set_model_state(model, state):
    if state is not None:
        model_cuda = adjust_model(state)
        model.load_state_dict(model_cuda)
    return model


@PtLogger()
def set_optimizer_state(optimizer, state):
    optimizer.load_state_dict(state)
    return optimizer


@PtLogger()
def load_parallel_model(model):
    if torch.cuda.is_available():
        device_ids = get_gpu_device_ids()
        if device_ids:
            device_ids = list(map(int, device_ids.split(",")))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model
    else:
        return model


@PtLogger(debug=True)
def adjust_model(state):
    # WhenEver a model is trained on multi gpu using DataParallel, module keyword is added
    model = {
        ([".".join(key.split(".")[1:])][0] if "module" in key.split(".")[0] else key): (
            value
        )
        for key, value in state.items()
    }
    return model
