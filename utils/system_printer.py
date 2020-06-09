import sys
import time

from utils.date_time_utility import get_time


class SystemPrinter:
    def __init__(self):
        self.start = time.time()

    def compute_eta(self, current_iter, total_iter):
        e = time.time() - self.start
        eta = e * total_iter / current_iter - e
        return get_time(eta)

    @staticmethod
    def dynamic_print(tag="Dynamic Print", data="Print"):
        log_format = " \033[1;37m>>\033[0m \033[93m{} \033[0m-\033[0m {}"
        sys.stdout.write("\r\x1b[K" + log_format.format(tag, data.__str__()))
        sys.stdout.flush()

    @staticmethod
    def sys_print(data: str):
        sys.stdout.writelines(data)
