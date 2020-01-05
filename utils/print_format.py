import traceback
from sys import stdout

from termcolor import colored
from tabulate import tabulate


def colored_dual_string_print_with_brackets(
    print_string_1: str,
    print_string_2: str,
    color_string_1: str,
    color_string_2: str,
    attrs=None,
):
    print(
        "[{}] - {}".format(
            colored("{}", color_string_1, attrs=attrs).format(print_string_1),
            colored(print_string_2, color_string_2),
        )
    )


def colored_single_string_print_with_brackets(
    print_string_1: str, color_string_1: str, attrs=None
):
    print(
        "[{}]".format(colored("{}", color_string_1, attrs=attrs).format(print_string_1))
    )


def colored_single_string_print(print_string_1: str, color_string_1: str, attrs=None):
    print(
        "{}".format(colored("{}", color_string_1, attrs=attrs).format(print_string_1))
    )


def colored_dual_string_print(
    print_string_1: str,
    print_string_2: str,
    color_string_1: str,
    color_string_2: str,
    attrs=None,
):
    print(
        "{} - {}".format(
            colored(print_string_1, color_string_1),
            colored(print_string_2, color_string_2),
            attrs=attrs,
        )
    )


def print_exception(exception: str, error_name: str, error_message: str):
    colored_dual_string_print_with_brackets(
        error_name, error_message, "red", "yellow", attrs=["blink"]
    )
    colored_dual_string_print_with_brackets(
        "Exception", str(exception), "red", "yellow", attrs=["blink"]
    )
    colored_dual_string_print_with_brackets(
        "Traceback", get_traceback(), "red", "yellow", attrs=["blink"]
    )


def get_traceback():
    return str(traceback.format_exc())


def tabular_print_handler(data: dict, headers: list):
    # For now can only handle only two columns
    tabular_data = []
    for key, value in data.items():
        tabular_data.append([key, "{:.5f}".format(value)])
    print(tabulate(tabular_data, headers=headers, tablefmt="orgtbl"))


def print_tab_fancy(print_string_1, color_string_1, attrs=None):
    print(
        "\t--{}".format(
            colored("{}", color_string_1, attrs=attrs).format(print_string_1)
        )
    )


def dual_print_tab_fancy(
    print_string_1, print_string_2, color_string_1, color_string_2
):
    colored_single_string_print_with_brackets(print_string_1, color_string_1)
    print_tab_fancy(print_string_2, color_string_2)


def print_dict(data):
    for key, value in data.items():
        if type(value) is dict:
            colored_single_string_print_with_brackets(key, "red")
            for sub_key, sub_value in value.items():
                print_tab_fancy("{}:{}".format(sub_key, sub_value), "yellow")
        else:
            dual_print_tab_fancy(key, value, "red", "yellow")
