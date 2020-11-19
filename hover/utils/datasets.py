"""
Submodule that loads and preprocesses public datasets into formats that work smoothly.
"""

from sklearn.datasets import fetch_20newsgroups
from hover import module_config
import re


def clean_string(text, sub_from=r"[^a-zA-Z0-9\ ]", sub_to=r" "):
    cleaned = re.sub(sub_from, sub_to, text)
    cleaned = re.sub(r" +", r" ", cleaned)
    return cleaned


def newsgroups_dictl(
    data_home="~/scikit_learn_data",
    to_remove=("headers", "footers", "quotes"),
    text_key="text",
    label_key="label",
    label_mapping=None,
):
    """
    Load the 20 Newsgroups dataset into a list of dicts, deterministically.
    """
    label_mapping = label_mapping or dict()
    dataset = dict()
    label_set = set()
    for _key in ["train", "test"]:
        _dictl = []

        # load subset and transform into a list of dicts
        _bunch = fetch_20newsgroups(
            data_home=data_home, subset=_key, random_state=42, remove=to_remove
        )
        for i, text in enumerate(_bunch.data):
            _text = clean_string(text)
            _label = _bunch.target_names[_bunch.target[i]]
            _label = label_mapping.get(_label, _label)

            _text_actual_characters = re.sub(r"[^a-zA-Z0-9]", r"", _text)
            if len(_text_actual_characters) > 5:
                label_set.add(_label)
                _entry = {text_key: _text, label_key: _label}
                _dictl.append(_entry)

        # add to dataset
        dataset[_key] = _dictl

    label_list = sorted(list(label_set))
    label_decoder = {idx: value for idx, value in enumerate(label_list)}
    label_decoder[module_config.ABSTAIN_ENCODED] = module_config.ABSTAIN_DECODED
    label_encoder = {value: idx for idx, value in label_decoder.items()}
    return dataset, label_encoder, label_decoder


def newsgroups_reduced_dictl(**kwargs):
    """
    Load the 20 Newsgroups dataset but reduce categories using a custom mapping.
    """
    label_mapping = {
        "alt.atheism": "religion",
        "comp.graphics": "computer",
        "comp.os.ms-windows.misc": "computer",
        "comp.sys.ibm.pc.hardware": "computer",
        "comp.sys.mac.hardware": "computer",
        "comp.windows.x": "computer",
        "misc.forsale": "forsale",
        "rec.autos": "recreation",
        "rec.motorcycles": "recreation",
        "rec.sport.baseball": "recreation",
        "rec.sport.hockey": "recreation",
        "sci.crypt": "computer",
        "sci.electronics": "computer",
        "sci.med": "med",
        "sci.space": "space",
        "soc.religion.christian": "religion",
        "talk.politics.guns": "politics",
        "talk.politics.mideast": "politics",
        "talk.politics.misc": "politics",
        "talk.religion.misc": "religion",
    }
    kwargs["label_mapping"] = label_mapping
    return newsgroups_dictl(**kwargs)
