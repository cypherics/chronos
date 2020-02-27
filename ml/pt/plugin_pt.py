class PluginPt(object):
    def __init__(self, plugin):
        self.model = self.get_model(plugin)
        self.loss = self.get_loss(plugin)
        self.validation = self.get_validation(plugin)
        self.train_data_set, self.val_data_set, self.test_data_set = self.get_data_set(plugin)

    @staticmethod
    def get_model(plugin):
        return plugin.load_model(plugin.get_model_module())

    @staticmethod
    def get_loss(plugin):
        return plugin.load_loss(plugin.get_loss_module())

    @staticmethod
    def get_validation(plugin):
        return plugin.load_validation(plugin.get_data_set_module())

    @staticmethod
    def get_data_set(plugin):
        return plugin.load_data_set(plugin.get_data_set_module())
