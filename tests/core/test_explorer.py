"""
Note that the whole point of explorers is to allow interaction, for which this file should not be considered a complete suite of tests.
"""

from hover import module_config
from hover.utils.snorkel_helper import labeling_function
import hover.core.explorer as hovex
import pytest
import random

PSEUDO_LABELS = ["A", "B"]


def RANDOM_LABEL(row):
    return random.choice(PSEUDO_LABELS)


def RANDOM_SCORE(row):
    return random.uniform(0.2, 1.0)


RANDOM_LABEL_LF = labeling_function(targets=PSEUDO_LABELS)(RANDOM_LABEL)


@pytest.fixture
def example_raw_df(generate_text_df_with_coords):
    df = generate_text_df_with_coords(300)
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
def example_dev_df(generate_text_df_with_coords):
    df = generate_text_df_with_coords(100)
    df["label"] = df.apply(RANDOM_LABEL, axis=1)
    return df


@pytest.mark.core
class TestBokehTextFinder:
    @staticmethod
    def test_comprehensive(example_raw_df, example_dev_df):
        """
        Some methods are the same across child classes.

        Test as many of those as possible here.
        """

        def subroutine(df_dict):
            explorer = hovex.BokehTextFinder(df_dict)
            annotator = hovex.BokehTextAnnotator(df_dict)

            explorer.reset_figure()

            explorer.plot()

            explorer.dfs["raw"] = example_dev_df
            explorer._update_sources()
            explorer.dfs["raw"] = example_raw_df
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
class TestBokehTextAnnotator:
    @staticmethod
    def test_annotation(example_raw_df):
        from bokeh.events import MenuItemClick

        explorer = hovex.BokehTextAnnotator({"raw": example_raw_df})
        explorer.plot()
        _ = explorer.view()

        explorer._callback_apply()

        for _item in ["Excel", "CSV", "JSON", "pickle"]:
            _event = MenuItemClick(explorer.annotator_export, item=_item)
            explorer._callback_export(_event)


@pytest.mark.core
class TestBokehTextSoftLabel:
    @staticmethod
    def test_init(example_soft_label_df):
        explorer = hovex.BokehTextSoftLabel(
            {"raw": example_soft_label_df, "train": example_soft_label_df.copy()},
            "pred_label",
            "pred_score",
        )
        explorer.plot()
        _ = explorer.view()


@pytest.mark.core
class TestBokehTextMargin:
    @staticmethod
    def test_init(example_margin_df):
        explorer = hovex.BokehTextMargin(
            {"raw": example_margin_df}, "label_1", "label_2"
        )
        explorer.plot("A")
        explorer.plot("B")
        _ = explorer.view()


@pytest.mark.core
class TestBokehTextSnorkel:
    @staticmethod
    def test_init(example_raw_df, example_dev_df):
        explorer = hovex.BokehTextSnorkel(
            {"raw": example_raw_df, "labeled": example_dev_df}
        )
        explorer.plot()
        explorer.plot_lf(RANDOM_LABEL_LF, include=("C", "I", "M", "H"))
        _ = explorer.view()
