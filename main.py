import fire

from state.default import Default
from state.resume import Resume


class Init(object):
    def __init__(self, config_path):
        self.default = Default(config_path)
        self.resume = Resume(config_path)


if __name__ == "__main__":
    fire.Fire(Init)
