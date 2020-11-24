from snorkel.labeling import labeling_function as snorkel_lf
from hover import module_params
import uuid


def labeling_function(targets, label_decoder, *args, **kwargs):
    """
    Hover flavor of the Snorkel labeling_function decorator.

    (1) assigns a UUID for easy identification;
    (2) keeps track of LF targets.

    :param targets: labels that the labeling function is intended to create.
    :type targets: list of int
    :param label_decoder: {encoded_label -> decoded_label} mapping.
    :type label_decoder: dict
    """

    def wrapper(func):
        lf = snorkel_lf(*args, **kwargs)(func)
        lf.uuid = uuid.uuid1()
        lf.targets = targets[:]
        lf.label_decoder = label_decoder
        lf.row_to_label = lambda row: lf.label_decoder.get(
            lf(row), module_params.ABSTAIN_DECODED
        )
        return lf

    return wrapper
