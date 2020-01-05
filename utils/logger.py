import logging
import os
import sys
import traceback

from utils import date_time_utility


class Logger:
    def __init__(self, folder_path, exp_name):
        log_path = os.path.join(folder_path, exp_name + ".log")
        self.logger = logging.getLogger("PyTrainer")
        self.logger.setLevel(logging.DEBUG)
        logger_handler = logging.FileHandler(log_path)

        logger_handler.setLevel(logging.DEBUG)

        logger_formatter = logging.Formatter(
            "%(asctime)s %(name)s - %(levelname)s - %(message)s"
        )
        logger_handler.setFormatter(logger_formatter)

        self.logger.addHandler(logger_handler)
        self.logger.info("Experiment : {}".format(exp_name))
        self.logger.info("DATE : {}".format(str(date_time_utility.get_date())))

    def log_info(self, msg):
        self.logger.info(msg)

    def log_exception(self, exception):
        msg = "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
            function_name=extract_function_name(),  # this is optional
            exception_class=exception.__class__,
            exception_docstring=exception.__doc__,
            exception_message=exception,
        )

        self.logger.exception(msg)


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
