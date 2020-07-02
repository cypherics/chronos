class BaseCallback:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_callbacks(self) -> list:
        raise NotImplementedError
