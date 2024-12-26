from typing import Any, Callable, List, Optional
import uuid


class LabelingFunction:
    """
    ???+ note "Function intended for labeling data points."
        The purpose is similar to [Snorkel's labeling function](https://github.com/snorkel-team/snorkel/blob/617c92400c50e95ce41fcee84309a86f76cf525c/snorkel/labeling/lf/core.py#L7), but with additional features:

        - the function must return an original string label, not its encoding integer
        - assigns a UUID for easy identification
        - keeps track of LF targets

        and removed features:
        - preprocessors and resources which need to be handled by the labeling function itself
    """
    def __init__(self, name: str, f: Callable[..., str], targets: List[str]):
        self.name = name
        self._f = f
        self.uuid = uuid.uuid1()
        self.targets = targets[:]

    def __call__(self, x: Any) -> str:
        return self._f(x)

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name}"


def labeling_function(
    targets: List[str],
    name: Optional[str] = None,
) -> Callable[[Callable[..., str]], LabelingFunction]:
    """
    ???+ note "Decorator that turns a function into a LabelingFunction object."

        | Param        | Type   | Description                          |
        | :----------- | :----- | :----------------------------------- |
        | `targets`    | `list` of `str` | labels that the labeling function is intended to create |
        | `name`       | `str` or `None` | name of the labeling function; defaults to `__name__` attribute |
    """

    def wrapper(func: Callable[..., str]) -> LabelingFunction:
        func_name = name or func.__name__
        lf = LabelingFunction(name=func_name, f=func, targets=targets)
        return lf

    return wrapper
