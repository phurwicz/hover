"""
Note that the whole point of explorers is to allow interaction, for which this file should not be considered a complete suite of tests.
"""

from hover import module_config
from hover.utils.snorkel_helper import labeling_function
from hover.recipes.subroutine import get_explorer_class
from bokeh.events import SelectionGeometry, MenuItemClick
import pytest
import random

MAIN_FEATURES = ["text", "image", "audio"]

PSEUDO_LABELS = ["A", "B"]


def RANDOM_LABEL(row):
    return random.choice(PSEUDO_LABELS)


def RANDOM_SCORE(row):
    return random.uniform(0.2, 1.0)


RANDOM_LABEL_LF = labeling_function(targets=PSEUDO_LABELS)(RANDOM_LABEL)


def almost_global_select(figure):
    select_event = SelectionGeometry(
        figure,
        geometry={
            "type": "poly",
            "x": [-1e4, -1e4, 1e4, 1e4],
            "y": [-1e4, 1e4, 1e4, -1e4],
            "sx": [None, None, None, None],
            "sy": [None, None, None, None],
        },
    )
    return select_event


def test_selection_filter(explorer, filter_toggle, narrowing_callbacks):
    total_raw = explorer.dfs["raw"].shape[0]
    initial_select = list(range(total_raw))

    # emulate user interface: select everything through a SelectionGeometry event
    explorer.sources["raw"].selected.indices = initial_select[:]
    # TODO: ideally should select by event, but this is not working
    # select_event = almost_global_select(explorer.figure)
    # explorer.figure._trigger_event(select_event)
    assert explorer.sources["raw"].selected.indices == initial_select[:]

    # trigger the first callback without activating filter
    narrowing_callbacks[0]()
    assert explorer.sources["raw"].selected.indices == initial_select

    # activate filter
    filter_toggle.active = [0]
    first_filter_select = explorer.sources["raw"].selected.indices[:]
    assert first_filter_select != initial_select
    assert set(first_filter_select).issubset(set(initial_select))

    # trigger the subsequent callbacks; selection should narrow
    _prev_select = first_filter_select[:]
    for _callback in narrowing_callbacks[1:]:
        _callback()
        _curr_select = explorer.sources["raw"].selected.indices[:]
        assert _curr_select != _prev_select
        assert set(_curr_select).issubset(set(_prev_select))

    # deactivate filter
    filter_toggle.active = []
    unfilter_select = explorer.sources["raw"].selected.indices[:]
    assert unfilter_select == initial_select


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

        def first_condition():
            explorer.search_pos.value = r"(?i)s[aeiou]\ "
            return

        def second_condition():
            explorer.search_neg.value = r"(?i)s[ae]\ "
            return

        test_selection_filter(
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

            _explorer.annotator_input.value = "A"
            _explorer.sources["raw"].selected.indices = [0, 2]
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

    @staticmethod
    def test_filter_score(example_soft_label_df):
        explorer = get_explorer_class("softlabel", "text")(
            {"raw": example_soft_label_df}
        )
        explorer.plot()

        def first_condition():
            explorer.score_range.value = (0.3, 0.8)
            return

        def second_condition():
            explorer.score_range.value = (0.5, 0.6)
            return

        test_selection_filter(
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

        # emulate selection by user
        # slice to first ten, then assign A to first six
        all_raw_idx = list(range(explorer.dfs["raw"].shape[0]))
        explorer.sources["raw"].selected.indices = all_raw_idx[:]
        # TODO: ideally should select by event, but this is not working
        # select_event = almost_global_select(explorer.figure)
        # explorer.figure._trigger_event(select_event)
        assert explorer.sources["raw"].selected.indices == all_raw_idx
        _event = MenuItemClick(explorer.lf_filter_trigger, item="broad_rule_a")
        explorer.lf_filter_trigger._trigger_event(_event)
        _event = MenuItemClick(explorer.lf_apply_trigger, item="narrow_rule_a")
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

        # slice to first ten, then assign B to first six
        explorer.sources["raw"].selected.indices = all_raw_idx[:]
        # TODO: ideally should select by event, but this is not working
        # explorer.figure._trigger_event(select_event)
        _event = MenuItemClick(explorer.lf_filter_trigger, item="broad_rule_b")
        explorer.lf_filter_trigger._trigger_event(_event)
        _event = MenuItemClick(explorer.lf_apply_trigger, item="narrow_rule_b")
        explorer.lf_apply_trigger._trigger_event(_event)

        first_six_labels = explorer.dfs["raw"]["label"].iloc[:6].tolist()
        assert first_six_labels == ["B"] * 6
