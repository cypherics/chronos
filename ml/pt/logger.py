import sys
import traceback

import os
import functools
import logging

from utils.date_time_utility import get_date


def extract_function_name():
    """Extracts failing function name from Traceback
    by Alex Martelli
    http://stackoverflow.com/questions/2380073/\
    how-to-identify-what-function-call-raise-an-exception-in-python
    """
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, 1)
    function_name = stk[0][3]
    return function_name


def create_logger(folder_path, exp_name):
    logger = logging.getLogger("PyTrainer-log")
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(folder_path, exp_name + ".log")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    log_format = " \033[1;37m>>\033[0m \033[93m[%(asctime)s][%(name)s][%(levelname)s] \033[0;37m-\033[0m %(message)s"
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fl = logging.FileHandler(log_path)
    fl.setLevel(logging.DEBUG)
    fl_format = logging.Formatter("%(asctime)s %(name)s : %(levelname)s : %(message)s")
    fl.setFormatter(fl_format)
    logger.addHandler(fl)
    logger.info("Experiment {} conducted on : {}".format(exp_name, get_date()))

    sys.stdout.write = logger.info


class PtLogger(object):
    def __init__(self, log_argument=False, log_result=False):
        self.logger = logging.getLogger("PyTrainer-log")
        self.log_argument = log_argument
        self.log_result = log_result

    def __call__(self, fn):
        @functools.wraps(fn)
        def log_decorated(*args, **kwargs):
            try:
                if self.log_argument:
                    self.logger.debug(
                        "{0} - {1} - {2}".format(fn.__name__, args, kwargs)
                    )
                else:
                    self.logger.info("{}".format(fn.__name__).upper())
                result = fn(*args, **kwargs)
                if self.log_result:
                    self.logger.debug("{0} - {1}".format(fn.__name__, result))
                return result
            except KeyboardInterrupt as ex:
                msg = "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
                    function_name=extract_function_name(),
                    exception_class=ex.__class__,
                    exception_docstring=ex.__doc__,
                    exception_message=ex,
                )

                self.logger.exception("Exception {0}".format(msg))
                raise ex
            except Exception as ex:
                msg = "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
                    function_name=extract_function_name(),
                    exception_class=ex.__class__,
                    exception_docstring=ex.__doc__,
                    exception_message=ex,
                )

                self.logger.exception("Exception {0}".format(msg))
                raise ex

        return log_decorated
