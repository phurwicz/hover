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
    # lazy import so that the package does not require snorkel
    # Feb 3, 2022: snorkel's dependency handling is too strict
    # for other dependencies like NumPy, SciPy, SpaCy, etc.
    # Let's cite Snorkel and lazy import or copy functions.
    # DO NOT explicitly depend on Snorkel without confirming
    # that all builds/tests pass by Anaconda standards, else
    # we risk having to drop conda support.
    from snorkel.labeling import (
        labeling_function as snorkel_lf,
        LabelingFunction as SnorkelLF,
    )

    def wrapper(func):
        # set up kwargs for Snorkel's LF
        # a default name that can be overridden
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
