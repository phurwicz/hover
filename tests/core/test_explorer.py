"""
Note that the whole point of explorers is to allow interaction, for which this file should not be considered a complete suite of tests.
"""

from hover.core.explorer import (
    BokehCorpusExplorer,
    BokehCorpusAnnotator,
    BokehSoftLabelExplorer,
    BokehMarginExplorer,
    BokehSnorkelExplorer,
)
from hover.utils.snorkel_helper import labeling_function
import pytest
import random

PSEUDO_LABELS = ["A", "B"]
RANDOM_LABEL = lambda x: random.choice(PSEUDO_LABELS)
RANDOM_SCORE = lambda x: random.uniform(0.2, 1.0)
RANDOM_LABEL_LF = labeling_function(targets=PSEUDO_LABELS)(RANDOM_LABEL)


@pytest.fixture
def example_raw_df(generate_text_df_with_coords):
    return generate_text_df_with_coords(300)


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
class TestBokehCorpusExplorer:
    @staticmethod
    def test_comprehensive(example_raw_df, example_dev_df):
        """
        Some methods are the same across child classes.

        Test as many of those as possible here.
        """
        explorer = BokehCorpusExplorer({"raw": example_raw_df})
        other = BokehCorpusAnnotator({"raw": example_raw_df})

        explorer.reset_figure()
        assert len(explorer.figure.renderers) == 0

        explorer.plot()
        assert len(explorer.figure.renderers) == 1

        explorer.dfs["raw"] = example_dev_df
        explorer._update_sources()
        explorer.dfs["raw"] = example_raw_df
        explorer._update_sources()

        explorer.link_selection("raw", other, "raw")
        explorer.link_xy_range(other)

        _ = explorer.view()


@pytest.mark.core
class TestBokehCorpusAnnotator:
    @staticmethod
    def test_annotation(example_raw_df):
        explorer = BokehCorpusAnnotator({"raw": example_raw_df})
        explorer.plot()
        _ = explorer.view()

        explorer._callback_apply()
        explorer._callback_export()


@pytest.mark.core
class TestBokehSoftLabelExplorer:
    @staticmethod
    def test_init(example_soft_label_df):
        explorer = BokehSoftLabelExplorer(
            {"raw": example_soft_label_df, "labeled": example_soft_label_df.copy()},
            "pred_label",
            "pred_score",
        )
        explorer.plot()
        _ = explorer.view()


@pytest.mark.core
class TestBokehMarginExplorer:
    @staticmethod
    def test_init(example_margin_df):
        explorer = BokehMarginExplorer({"raw": example_margin_df}, "label_1", "label_2")
        explorer.plot("A")
        explorer.plot("B")
        _ = explorer.view()


@pytest.mark.core
class TestBokehSnorkelExplorer:
    @staticmethod
    def test_init(example_raw_df, example_dev_df):
        explorer = BokehSnorkelExplorer(
            {"raw": example_raw_df, "labeled": example_dev_df}
        )
        explorer.plot()
        explorer.plot_lf(RANDOM_LABEL_LF, include=("C", "I", "M", "H"))
        _ = explorer.view()
