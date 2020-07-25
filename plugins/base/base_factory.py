class Factory:
    def __init__(self, config):
        self.config = config

    def create_data_set(self):
        raise NotImplementedError

    def create_network(self, model_name, model_param):
        raise NotImplementedError

    def create_criterion(self, criterion_name, criterion_param):
        raise NotImplementedError

    def create_extension(self):
        raise NotImplementedError
