import fire

from instance.training import Training


class Init(object):
    def __init__(self, config_path):
        self.training = Training(config_path)


if __name__ == "__main__":
    fire.Fire(Init)
