import os


class BaseExtension:
    def __init__(self, config):
        self.config = config
        self.pth = os.path.join(
            os.path.join(config.training_path, config.version), "extension"
        )

    def callbacks(self) -> list:
        raise NotImplementedError

    def metrics(self) -> list:
        raise NotImplementedError
