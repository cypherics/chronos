import os

from plugins.base.base_factory import Factory
from core.logger import info
from utils.system_printer import SystemPrinter


class PtPlugin:
    def __init__(self, config):
        self.config = config
        self.factory = self.create_factory()

    @info
    def create_factory(self):

        plugin_name = self.config.plugin
        plugin = os.path.join("plugins/", plugin_name)

        file = os.path.join(os.curdir, plugin, plugin_name + "_factory.py")

        import importlib.util

        spec = importlib.util.spec_from_file_location(plugin_name, file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        for cls in Factory.__subclasses__():
            f = cls(self.config)
            SystemPrinter.sys_print(
                "\t LOADED PLUGIN FACTORY - {}".format(f.__class__.__name__)
            )
            return f

    def load_plugin(self, config):
        model = self.factory.create_network(config.model_name, config.model_param)
        SystemPrinter.sys_print("\t LOADED MODEL - {}".format(model.__class__.__name__))

        criterion = self.factory.create_criterion(config.loss_name, config.loss_param)
        SystemPrinter.sys_print(
            "\t LOADED CRITERION - {}".format(criterion.__class__.__name__)
        )

        evaluator = self.factory.create_evaluator()
        data_loader = self.factory.create_data_set()
        plugin_callbacks = self.factory.create_callback()

        return model, criterion, evaluator, data_loader
