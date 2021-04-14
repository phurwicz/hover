"""
Note that the whole point of explorers is to allow interaction, for which this file should not be considered a complete suite of tests.
"""

from hover import module_config
from hover.utils.snorkel_helper import labeling_function
from hover.recipes.subroutine import get_explorer_class
import pytest
import random

MAIN_FEATURES = ["text", "image", "audio"]

PSEUDO_LABELS = ["A", "B"]


def RANDOM_LABEL(row):
    return random.choice(PSEUDO_LABELS)


def RANDOM_SCORE(row):
    return random.uniform(0.2, 1.0)


RANDOM_LABEL_LF = labeling_function(targets=PSEUDO_LABELS)(RANDOM_LABEL)


@pytest.fixture
def example_raw_df(generate_df_with_coords):
    df = generate_df_with_coords(300)
    df["label"] = module_config.ABSTAIN_DECODED
    return df


@pytest.fixture
def example_soft_label_df(example_raw_df):
    df = example_raw_df.copy()
    df["pred_label"] = df.apply(RANDOM_LABEL, axis=1)
    df["pred_score"] = df.apply(RANDOM_SCORE, axis=1)
    return df


@pytest.fixture
def example_margin_df(example_raw_df):
    df = example_raw_df.copy()
    df["label_1"] = df.apply(RANDOM_LABEL, axis=1)
    df["label_2"] = df.apply(RANDOM_LABEL, axis=1)
    return df


@pytest.fixture
def example_dev_df(generate_df_with_coords):
    df = generate_df_with_coords(100)
    df["label"] = df.apply(RANDOM_LABEL, axis=1)
    return df


@pytest.mark.core
class TestBokehBaseExplorer:
    @staticmethod
    def test_comprehensive(example_raw_df, example_dev_df):
        """
        Some methods are the same across child classes.

        Test as many of those as possible here.
        """

        def subroutine(df_dict):
            explorer = get_explorer_class("finder", "text")(df_dict)
            annotator = get_explorer_class("annotator", "text")(df_dict)

            explorer.plot()

            explorer.dfs["raw"] = example_raw_df.copy()
            explorer._update_sources()

            explorer.link_selection("raw", annotator, "raw")
            explorer.link_xy_range(annotator)

            _ = explorer.view()

        df_dict = {
            "raw": example_raw_df.copy(),
            "train": example_dev_df.copy(),
            "dev": example_dev_df.copy(),
            "test": example_dev_df.copy(),
        }

        for _key in ["test", "dev", "train", "raw"]:
            subroutine(df_dict)
            df_dict.pop(_key)


@pytest.mark.core
class TestBokehDataAnnotator:
    @staticmethod
    def test_annotation(example_raw_df):
        # test most methods for the class corresponding to each kind of feature
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("annotator", _feature)
            _explorer = _cls({"raw": example_raw_df})
            _explorer.plot()
            _ = _explorer.view()

            _explorer._callback_apply()


@pytest.mark.core
class TestBokehTextSoftLabel:
    @staticmethod
    def test_init(example_soft_label_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("softlabel", _feature)
            _explorer = _cls(
                {"raw": example_soft_label_df, "train": example_soft_label_df.copy()},
                "pred_label",
                "pred_score",
            )
            _explorer.plot()
            _ = _explorer.view()


@pytest.mark.core
class TestBokehTextMargin:
    @staticmethod
    def test_init(example_margin_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("margin", _feature)
            _explorer = _cls({"raw": example_margin_df}, "label_1", "label_2")
            _explorer.plot("A")
            _explorer.plot("B")
            _ = _explorer.view()


@pytest.mark.core
class TestBokehTextSnorkel:
    @staticmethod
    def test_init(example_raw_df, example_dev_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("snorkel", _feature)
            _explorer = _cls({"raw": example_raw_df, "labeled": example_dev_df})
            _explorer.plot()
            _explorer.plot_lf(RANDOM_LABEL_LF, include=("C", "I", "M", "H"))
            _ = _explorer.view()
