import os

from ml.ml_type.base.base_factory import Factory
from ml.pt.logger import info
from utils.system_printer import SystemPrinter


class PtPlugin:
    def __init__(self, config):
        self.config = config
        self.factory = self.create_factory()

    @info
    def create_factory(self):

        plugin_name = self.config.plugin
        plugin = os.path.join("ml/ml_type/", plugin_name)

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
