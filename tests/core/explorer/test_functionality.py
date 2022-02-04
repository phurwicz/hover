"""
Note that the whole point of explorers is to allow interaction.
Therefore we need to emulate user interaction through bokeh.events objects.
"""

from hover import module_config
from hover.utils.snorkel_helper import labeling_function
from hover.recipes.subroutine import get_explorer_class
from bokeh.events import ButtonClick, MenuItemClick
from .local_helper import (
    MAIN_FEATURES,
    RANDOM_LABEL,
    RANDOM_SCORE,
    RANDOM_LABEL_LF,
    almost_global_select,
    subroutine_selection_filter,
)
import pytest
import random
import re


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
class TestBokehDataFinder:
    @staticmethod
    def test_init(example_raw_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("finder", _feature)
            _explorer = _cls({"raw": example_raw_df})
            _explorer.plot()
            _ = _explorer.view()

    @staticmethod
    def test_filter_text(example_raw_df):
        explorer = get_explorer_class("finder", "text")({"raw": example_raw_df})
        explorer.plot()

        # dynamically construct patterns with predictable outcome
        texts = explorer.dfs["raw"]["text"].tolist()
        first_token_of_ten = set()
        first_token_of_two = set()
        for i, _text in enumerate(texts):
            # terminate when the two sets are different
            if i >= 10 and first_token_of_ten != first_token_of_two:
                assert first_token_of_two.issubset(first_token_of_ten)
                break

            _tokens = _text.split(" ")
            # guard against empty text case
            if not _tokens:
                continue

            # add token to sets
            first_token_of_ten.add(_tokens[0])
            if i < 2:
                first_token_of_two.add(_tokens[0])

        broad_pattern = "(?i)(" + "|".join(first_token_of_ten) + ")"
        narrow_pattern = "(?i)(" + "|".join(first_token_of_two) + ")"
        assert re.search(narrow_pattern, texts[0])
        assert re.search(broad_pattern, texts[2])

        def first_condition():
            explorer.search_pos.value = broad_pattern
            return

        def second_condition():
            explorer.search_neg.value = narrow_pattern
            return

        subroutine_selection_filter(
            explorer,
            explorer.search_filter_box,
            [first_condition, second_condition],
        )


@pytest.mark.core
class TestBokehDataAnnotator:
    @staticmethod
    def test_init(example_raw_df):
        # test most methods for the class corresponding to each kind of feature
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("annotator", _feature)
            _explorer = _cls({"raw": example_raw_df})
            _explorer.plot()
            _ = _explorer.view()

    @staticmethod
    def test_labeling(example_raw_df):
        feature = random.choice(MAIN_FEATURES)
        explorer = get_explorer_class("annotator", feature)({"raw": example_raw_df})

        # empty click
        apply_event = ButtonClick(explorer.annotator_apply)
        explorer.annotator_apply._trigger_event(apply_event)

        # test non-cumulative selection
        explorer.sources["raw"].selected.indices = [0]
        explorer._store_selection()
        assert explorer.sources["raw"].selected.indices == [0]
        explorer.sources["raw"].selected.indices = [1]
        explorer._store_selection()
        assert explorer.sources["raw"].selected.indices == [1]

        # test cumulative selection
        explorer.selection_option_box.active = [0]
        explorer.sources["raw"].selected.indices = [0]
        explorer._store_selection()
        assert explorer.sources["raw"].selected.indices == [0, 1]

        # actual labeling
        explorer.annotator_input.value = "A"
        explorer.annotator_apply._trigger_event(apply_event)


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

    @staticmethod
    def test_filter_score(example_soft_label_df):
        explorer = get_explorer_class("softlabel", "text")(
            {"raw": example_soft_label_df},
            "pred_label",
            "pred_score",
        )
        explorer.plot()

        def first_condition():
            explorer.score_range.value = (0.3, 0.8)
            return

        def second_condition():
            explorer.score_range.value = (0.5, 0.6)
            return

        subroutine_selection_filter(
            explorer,
            explorer.score_filter_box,
            [first_condition, second_condition],
        )


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

    @staticmethod
    def test_lf_labeling(example_raw_df, example_dev_df):
        explorer = get_explorer_class("snorkel", "text")(
            {
                "raw": example_raw_df,
                "labeled": example_dev_df,
            }
        )
        explorer.plot()

        # create some dummy rules for predictable outcome
        texts = explorer.dfs["raw"]["text"].tolist()
        first_six_texts = set(texts[:6])
        first_ten_texts = set(texts[:10])

        @labeling_function(targets=["A"])
        def narrow_rule_a(row):
            if row["text"] in first_six_texts:
                return "A"
            return module_config.ABSTAIN_DECODED

        @labeling_function(targets=["A"])
        def broad_rule_a(row):
            if row["text"] in first_ten_texts:
                return "A"
            return module_config.ABSTAIN_DECODED

        @labeling_function(targets=["B"])
        def narrow_rule_b(row):
            if row["text"] in first_six_texts:
                return "B"
            return module_config.ABSTAIN_DECODED

        @labeling_function(targets=["B"])
        def broad_rule_b(row):
            if row["text"] in first_ten_texts:
                return "B"
            return module_config.ABSTAIN_DECODED

        # add two rules, check menu
        explorer.plot_lf(narrow_rule_a)
        explorer.plot_lf(broad_rule_a)
        lf_names_so_far = ["narrow_rule_a", "broad_rule_a"]
        assert explorer.lf_apply_trigger.menu == lf_names_so_far
        assert explorer.lf_filter_trigger.menu == lf_names_so_far

        # add an existing rules, should trigger overwrite
        explorer.plot_lf(narrow_rule_a)
        assert explorer.lf_apply_trigger.menu == lf_names_so_far
        assert explorer.lf_filter_trigger.menu == lf_names_so_far

        # empty click: nothing selected
        _event = MenuItemClick(explorer.lf_filter_trigger, item="broad_rule_a")
        _event = MenuItemClick(explorer.lf_apply_trigger, item="narrow_rule_a")
        explorer.lf_filter_trigger._trigger_event(_event)
        explorer.lf_apply_trigger._trigger_event(_event)

        # emulate selection by user
        # slice to first ten, then assign A to first six
        all_raw_idx = list(range(explorer.dfs["raw"].shape[0]))
        # note: bokeh's SelectionGeometry seems to not actually make a selection
        # note: the indices assignment has to happen before SelectionGeometry trigger
        # - this is for treating indices assignment as a manual select
        explorer.sources["raw"].selected.indices = all_raw_idx[:]
        select_event = almost_global_select(explorer.figure)
        explorer.figure._trigger_event(select_event)
        assert explorer.sources["raw"].selected.indices == all_raw_idx

        # actually triggering LFs on a valid selection
        explorer.lf_filter_trigger._trigger_event(_event)
        explorer.lf_apply_trigger._trigger_event(_event)

        first_six_labels = explorer.dfs["raw"]["label"].iloc[:6].tolist()
        assert first_six_labels == ["A"] * 6

        # add more rules, check menu again
        explorer.plot_lf(narrow_rule_b)
        explorer.plot_lf(broad_rule_b)

        lf_names_so_far = [
            "narrow_rule_a",
            "broad_rule_a",
            "narrow_rule_b",
            "broad_rule_b",
        ]
        assert explorer.lf_apply_trigger.menu == lf_names_so_far
        assert explorer.lf_filter_trigger.menu == lf_names_so_far

        # note: bokeh's SelectionGeometry seems to not actually make a selection
        # note: the indices assignment has to happen before SelectionGeometry trigger
        # - this is for treating indices assignment as a manual select
        explorer.sources["raw"].selected.indices = all_raw_idx[:]
        explorer.figure._trigger_event(select_event)
        # slice to first ten, then assign B to first six
        _event = MenuItemClick(explorer.lf_filter_trigger, item="broad_rule_b")
        explorer.lf_filter_trigger._trigger_event(_event)
        _event = MenuItemClick(explorer.lf_apply_trigger, item="narrow_rule_b")
        explorer.lf_apply_trigger._trigger_event(_event)

        first_six_labels = explorer.dfs["raw"]["label"].iloc[:6].tolist()
        assert first_six_labels == ["B"] * 6