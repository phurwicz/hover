from snorkel.labeling import (
    labeling_function as snorkel_lf,
    LabelingFunction as SnorkelLF,
)
import uuid


def labeling_function(targets, label_encoder=None, **kwargs):
    """
    ???+ note "Hover's flavor of the Snorkel labeling_function decorator."
        However, due to the dynamic label encoding nature of hover,
        the decorated function should return the original string label, not its encoding integer.

        - assigns a UUID for easy identification
        - keeps track of LF targets

        | Param           | Type   | Description                          |
        | :-------------- | :----- | :----------------------------------- |
        | `targets`       | `list` of `str` | labels that the labeling function is intended to create |
        | `label_encoder` | `dict` | {decoded_label -> encoded_label} mapping, if you also want an original snorkel-style labeling function linked as a `.snorkel` attribute |
        | `**kwargs`      |        | forwarded to `snorkel`'s `labeling_function()` |
    """

    def wrapper(func):
        # set up kwargs for Snorkel's LF
        snorkel_kwargs = {"name": func.__name__}
        snorkel_kwargs.update(kwargs)

        # return value of hover's decorator
        lf = SnorkelLF(f=func, **snorkel_kwargs)

        # additional attributes
        lf.uuid = uuid.uuid1()
        lf.targets = targets[:]

        # link a snorkel-style labeling function if applicable
        if label_encoder:
            lf.label_encoder = label_encoder

            def snorkel_style_func(x):
                return lf.label_encoder[func(x)]

            lf.snorkel = snorkel_lf(**kwargs)(snorkel_style_func)
        else:
            lf.label_encoder = None
            lf.snorkel = None

        return lf

    return wrapper
