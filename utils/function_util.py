import sys
import traceback


def is_overridden_func(func):
    # https://stackoverflow.com/questions/9436681/how-to-detect-method-overloading-in-subclasses-in-python
    obj = func.__self__
    base_class = getattr(super(type(obj), obj), func.__name__)
    return func.__func__ != base_class.__func__


def extract_detail():
    """Extracts failing function name from Traceback
    by Alex Martelli
    http://stackoverflow.com/questions/2380073/how-to-identify-what-function-call-raise-an-exception-in-python
    """
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, -1)[0]
    return "{} in {} line num {} on line {} ".format(
        stk.name, stk.filename, stk.lineno, stk.line
    )


def get_details(fn):
    class_name = vars(sys.modules[fn.__module__])[
        fn.__qualname__.split(".")[0]
    ].__name__
    fn_name = fn.__name__
    if class_name == fn_name:
        return None, fn_name
    else:
        return class_name, fn_name
