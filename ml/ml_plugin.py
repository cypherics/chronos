import importlib
from utils.logger import LogDecorator


class PluginCollection(object):
    """Upon creation, this class will read the plugins package for modules
    that contain a class definition that is inheriting from the Plugin class
    """
    @LogDecorator()
    def __init__(self, plugin_package, training_configuration):
        """Constructor that initiates the reading of all available plugins
        when an instance of the PluginCollection object is created
        """
        self.plugin_package = plugin_package
        self.training_configuration = training_configuration
        module_name = "ml.ml_type." + self.plugin_package
        importlib.import_module(module_name)

    def get_plugins_function(self):
        """Reset the list of all plugins and initiate the walk over the main
        provided plugin package to load all available plugins
        """
        model = self.load_model(self.get_model_module())
        loss = self.load_loss(self.get_loss_module())

        validation = self.load_validation(self.get_data_set_module())
        train_data_set, val_data_set, test_data_set = self.load_data_set(self.get_data_set_module())

        return model, loss, validation, train_data_set, val_data_set, test_data_set

    @LogDecorator()
    def get_model_module(self):
        module_name = "ml.ml_type." + self.plugin_package
        importlib.import_module(module_name)
        model_package = importlib.import_module(".network", package=module_name)

        return model_package

    @LogDecorator()
    def get_data_set_module(self):
        module_name = "ml.ml_type." + self.plugin_package
        importlib.import_module(module_name)
        data_set_package = importlib.import_module(".dataset", package=module_name)

        return data_set_package

    @LogDecorator()
    def get_loss_module(self):
        module_name = "ml.ml_type." + self.plugin_package
        importlib.import_module(module_name)
        loss_package = importlib.import_module(".loss", package=module_name)

        return loss_package

    @LogDecorator()
    def load_model(self, model_package):
        model_name = self.training_configuration["model"]
        model_param = self.training_configuration[
            self.training_configuration["model"]
        ]
        model = getattr(model_package, model_name)(**model_param)

        return model

    @LogDecorator()
    def load_loss(self, loss_package):
        loss_name = self.training_configuration["loss"]
        loss_param = self.training_configuration[loss_name]
        loss = getattr(loss_package, loss_name)(**loss_param)

        return loss

    @LogDecorator()
    def load_validation(self, data_set_package):
        validation = getattr(data_set_package, "Validation")()
        return validation

    @LogDecorator()
    def load_data_set(self, data_set_package):
        root = self.training_configuration["root"]
        model_input_dimension = self.training_configuration["initial_assignment"][
            "model_input_dimension"
        ]
        normalization = self.training_configuration["normalization"]
        transformation = self.training_configuration["transformation"]
        train_data_set = getattr(data_set_package, "Dataloader")(
            root, model_input_dimension, "train", transformation, normalization
        )

        val_data_set = getattr(data_set_package, "Dataloader")(
            root, model_input_dimension, "val", transformation, normalization
        )

        test_data_set = getattr(data_set_package, "Dataloader")(
            root, model_input_dimension, "test", transformation, normalization
        )

        return train_data_set, val_data_set, test_data_set
