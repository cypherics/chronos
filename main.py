import fire

from train import Train


class Init(object):
    def __init__(self, plugin, config_path):
        self.train = Train(plugin, config_path)


if __name__ == "__main__":
    fire.Fire(Init)
