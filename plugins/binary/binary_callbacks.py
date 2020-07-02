from plugins.base.base_callbacks import BaseCallback


class BinaryCallback(BaseCallback):
    def __init__(self, config):
        super().__init__(config)

    def get_callbacks(self) -> list:
        pass
