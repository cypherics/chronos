from scipy import misc


def load_image(path: str):
    img = misc.imread(path)
    return img


def load_mask(path: str):
    mask = misc.imread(path)
    return mask
