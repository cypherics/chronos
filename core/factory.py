import os

from plugins.base.base_factory import Factory
from core.logger import info
from utils.system_printer import SystemPrinter


class Plugin:
    def __init__(self, config):
        self.config = config
        self.factory = self.create_factory()
        self.model = None
        self.criterion = None
        self.loader = None
        self.extension = None

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

    def load_plugin(self):
        self.model = self.factory.create_network(
            self.config.model_name, self.config.model_param
        )
        SystemPrinter.sys_print(
            "\t LOADED MODEL - {}".format(self.model.__class__.__name__)
        )

        self.criterion = self.factory.create_criterion(
            self.config.loss_name, self.config.loss_param
        )
        SystemPrinter.sys_print(
            "\t LOADED CRITERION - {}".format(self.criterion.__class__.__name__)
        )

        self.loader = self.factory.create_data_set()
        self.extension = self.factory.create_extension()
