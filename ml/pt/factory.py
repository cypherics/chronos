import os

from ml.ml_type.base.base_factory import Factory
from ml.pt.logger import PtLogger


class Plugin:
    def __init__(self, config):
        self.config = config
        self.factory = self.create_factory()

    @PtLogger()
    def create_factory(self):

        plugin_name = self.config.problem_type
        plugin = os.path.join("ml/ml_type/", plugin_name)

        file = os.path.join(os.curdir, plugin, plugin_name + "_factory.py")

        import importlib.util

        spec = importlib.util.spec_from_file_location(self.config.problem_type, file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        for cls in Factory.__subclasses__():
            f = cls(self.config)
            return f
