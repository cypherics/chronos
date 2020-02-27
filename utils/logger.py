import sys
import traceback

import os
import functools
import logging


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
    logger = logging.getLogger('PyTrainer-log')
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(folder_path, exp_name + ".log")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fl = logging.FileHandler(log_path)
    fl.setLevel(logging.DEBUG)
    fl_format = logging.Formatter(
        "%(asctime)s %(name)s - %(levelname)s - %(message)s"
    )
    fl.setFormatter(fl_format)
    logger.addHandler(fl)
    logger.info("Experiment : {}".format(exp_name))
    sys.stdout.write = logger.info


class LogDecorator(object):
    def __init__(self):
        self.logger = logging.getLogger('PyTrainer-log')

    def __call__(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            try:
                self.logger.debug("{0}".format(fn.__name__))
                result = fn(*args, **kwargs)
                self.logger.debug(result)
                return result
            except Exception as ex:
                msg = "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
                    function_name=extract_function_name(),  # this is optional
                    exception_class=ex.__class__,
                    exception_docstring=ex.__doc__,
                    exception_message=ex,
                )

                self.logger.exception("Exception {0}".format(msg))
                raise ex
        return decorated

