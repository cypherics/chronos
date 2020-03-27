import sys


class SystemPrinter:
    @staticmethod
    def dynamic_print(tag="Dynamic Print", data="Print"):
        log_format = " \033[1;37m>>\033[0m \033[93m{} \033[0m-\033[0m {}"
        sys.stdout.write("\r\x1b[K" + log_format.format(tag, data.__str__()))
        sys.stdout.flush()

    @staticmethod
    def sys_print(data):
        sys.stdout.writelines([data])
