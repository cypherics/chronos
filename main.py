import fire

from ml.train import Train


class Init(object):
    def __init__(self, config_path):
        self.train = Train(config_path)


if __name__ == "__main__":
    fire.Fire(Init)
