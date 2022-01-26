from abc import ABCMeta
from functools import wraps
from types import FunctionType
from rich.console import Console
from rich.prompt import Confirm


class RichTracebackMeta(type):
    """
    ???+ note "Metaclass to mass-add traceback override to class methods."

        [`Rich` traceback guide](https://rich.readthedocs.io/en/stable/traceback.html)
    """

    def __new__(meta, class_name, bases, class_dict):
        # prefers the class's CONSOLE attribute; create one otherwise
        console = class_dict.get("CONSOLE", Console())

        def wrapper(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    console.print(
                        f":red_circle: {func.__module__}.{func.__qualname__} failed.",
                        style="red bold",
                    )
                    locals_flag = Confirm.ask("Show local variables in traceback?")
                    console.print_exception(show_locals=locals_flag)

            return wrapped

        new_class_dict = {}
        for attr_name, attr_value in class_dict.items():
            # replace each method with a wrapped version
            if isinstance(attr_value, FunctionType):
                attr_value = wrapper(attr_value)
            new_class_dict[attr_name] = attr_value
        return type.__new__(meta, class_name, bases, new_class_dict)


class RichTracebackABCMeta(RichTracebackMeta, ABCMeta):
    """
    ???+ note "Metaclass for rich-traceback abstract base classes."

        To resolve the metaclass conflict between RichTracebackMeta and ABCMeta.
    """

    pass
