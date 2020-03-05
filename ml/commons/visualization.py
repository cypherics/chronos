import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from torch.utils.tensorboard import SummaryWriter

from utils.directory_handler import make_directory


class Visualization:
    def __init__(self, log_dir):
        log_dir = make_directory(log_dir, "events")
        self.writer = SummaryWriter(log_dir)

    def plt_scalar(self, y, x, tag):
        if type(y) is dict:
            self.writer.add_scalars(tag, y, global_step=x)
            self.writer.flush()
        else:
            self.writer.add_scalar(tag, y, global_step=x)
            self.writer.flush()

    def plt_images(self, img, global_step, tag):
        self.writer.add_image(tag, img, global_step)
        self.writer.flush()
