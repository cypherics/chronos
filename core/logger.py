import sys
import os
import functools
import logging
from logging.handlers import TimedRotatingFileHandler

from utils.date_time import get_date
from utils.directory_ops import make_directory
from utils.function_util import extract_detail, get_details


class ChronosLogger:
    def __init__(self):
        pass

    @staticmethod
    def get_logger():
        return logging.getLogger("Chronos-log")

    @staticmethod
    def create_channel_log(logger):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        log_format = (
            " \033[1;37m>>\033[0m \033[93m[%(asctime)s][%(name)s][%(levelname)s] \033[0;37m-"
            "\033[0m %(message)s"
        )
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    @staticmethod
    def create_time_rotated_log(log_path, plugin, exp_name, model, version, logger):
        extensive_log_file = os.path.join(log_path, exp_name + ".log")
        tfl = TimedRotatingFileHandler(extensive_log_file, when="D")
        tfl.setLevel(logging.DEBUG)
        rfl_format = logging.Formatter(
            "%(asctime)s %(name)s : %(levelname)-5s : Plugin: {:5} : ExpName: {:5} : Model: {:5} : Version: {:5} "
            ": %(message)s".format(plugin, exp_name, model, version)
        )
        tfl.setFormatter(rfl_format)
        logger.addHandler(tfl)

    @staticmethod
    def create_file_log(log_path, plugin, exp_name, model, version, logger):
        log_file = os.path.join(log_path, "{}-console".format(exp_name) + ".log")
        fl = logging.FileHandler(log_file)
        fl.setLevel(logging.INFO)
        fl_format = logging.Formatter(
            "%(asctime)s %(name)s : %(levelname)-5s : Plugin: {:5} : ExpName: {:5} : Model: {:5} : Version: {:5} "
            ": %(message)s".format(plugin, exp_name, model, version)
        )
        fl.setFormatter(fl_format)
        logger.addHandler(fl)

    def create_logger(self, folder_path, plugin, exp_name, model, version):
        logger = self.get_logger()
        logger.setLevel(logging.DEBUG)
        log_path = make_directory(folder_path, "logs")
        self.create_channel_log(logger)
        self.create_file_log(log_path, plugin, exp_name, model, version, logger)
        self.create_time_rotated_log(log_path, plugin, exp_name, model, version, logger)
        logger.info("Experiment {} conducted on : {}".format(exp_name, get_date()))

        sys.stdout.writelines = logger.info


def exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = ChronosLogger.get_logger()
        try:
            result = func(*args, **kwargs)
            return result
        except KeyboardInterrupt as ex:
            msg = "{detail} raised {exception_class} ({exception_docstring}): {exception_message}".format(
                detail=extract_detail(),
                exception_class=ex.__class__,
                exception_docstring=ex.__doc__,
                exception_message=ex,
            )
            msg_str = "Message : %s " % msg
            logger.exception(msg_str)
            raise ex
        except Exception as ex:
            msg = "{detail} raised {exception_class} ({exception_docstring}): {exception_message}".format(
                detail=extract_detail(),
                exception_class=ex.__class__,
                exception_docstring=ex.__doc__,
                exception_message=ex,
            )
            msg_str = "Message : %s " % msg
            logger.exception(msg_str)
            raise ex

    return wrapper


def info(func):
    @functools.wraps(func)
    @exception
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


def debug(func):
    @functools.wraps(func)
    @exception
    def wrapper(*args, **kwargs):

        logger = ChronosLogger.get_logger()
        class_name, func_name = get_details(func)
        func_args = args
        func_kwargs = kwargs

        if class_name is not None:
            msg_str = "Class - {:2} : Function - {:3} : args - {:3} : kwargs - {}".format(
                class_name, str(func_name), str(func_args), str(func_kwargs)
            )
        else:
            msg_str = "Function - {:3} : args - {:3} : kwargs - {}".format(
                func_name, str(func_args), str(func_kwargs)
            )

        logger.debug(msg_str)
        result = func(*args, **kwargs)
        if result is not None:
            msg_str = "Function - {:3} : Result - {}".format(func_name, result)
            logger.debug(msg_str)

        return result

    return wrapper
