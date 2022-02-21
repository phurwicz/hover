"""
Corresponds to the `hover.core.explorer.functionality` module.
For mechanisms that are invariant across `hover.core.explorer.feature`.
"""

from hover import module_config
from hover.recipes.subroutine import get_explorer_class
from bokeh.events import ButtonClick, MenuItemClick
from .local_helper import (
    MAIN_FEATURES,
    RANDOM_LABEL_LF,
    almost_global_select,
    subroutine_selection_filter,
    subroutine_rules_from_text_df,
)
import pytest
import random
import re


@pytest.mark.core
class TestBokehBaseExplorer:
    @staticmethod
    @pytest.mark.lite
    def test_comprehensive(example_raw_df, example_labeled_df):
        """
        Some methods are the same across child classes.

        Test as many of those as possible here.
        """

        def subroutine(df_dict, feature_type):
            finder = get_explorer_class("finder", feature_type)(df_dict)
            annotator = get_explorer_class("annotator", feature_type)(df_dict)

            finder.plot()
            annotator.plot()

            finder.dfs["raw"] = example_raw_df.copy()
            finder._update_sources()

            finder.link_selection("raw", annotator, "raw")
            finder.link_selection_options(annotator)
            finder.link_xy_range(annotator)

            # change axes and check source and glyph
            # the source should not change, just glyph axes
            embed_cols = annotator.find_embedding_fields()
            prev_source_data = annotator.sources["raw"].data.copy()
            inspect_renderer = annotator.figure.renderers[0]

            for _axis, _from_idx, _to_idx in [
                ("x", 0, 1),
                ("y", 1, 0),
            ]:
                _widget = getattr(annotator, f"dropdown_{_axis}_axis")
                _click_event = MenuItemClick(_widget, item=embed_cols[_to_idx])
                assert getattr(inspect_renderer.glyph, _axis) == embed_cols[_from_idx]
                _widget._trigger_event(_click_event)
                assert getattr(inspect_renderer.glyph, _axis) == embed_cols[_to_idx]
                assert annotator.sources["raw"].data is not prev_source_data
                assert annotator.sources["raw"].data == prev_source_data

        df_dict = {
            "raw": example_raw_df.copy(),
            "train": example_labeled_df.copy(),
            "dev": example_labeled_df.copy(),
            "test": example_labeled_df.copy(),
        }

        # test four subset combinations of df_dicts
        for _key in ["test", "dev", "train", "raw"]:
            # test all main features
            for _feature in MAIN_FEATURES:
                subroutine(df_dict, _feature)
            df_dict.pop(_key)


@pytest.mark.core
class TestBokehFinder:
    @staticmethod
    @pytest.mark.lite
    def test_init(example_raw_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("finder", _feature)
            _explorer = _cls({"raw": example_raw_df})
            _explorer.plot()
            _ = _explorer.view()

    @staticmethod
    @pytest.mark.lite
    def test_filter_text(example_raw_df):
        """
        Filtration given search has no difference across text / image / audio.
        Use text here for an example.
        """
        explorer = get_explorer_class("finder", "text")({"raw": example_raw_df})
        explorer.activate_search()
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
class TestBokehAnnotator:
    @staticmethod
    @pytest.mark.lite
    def test_init(example_raw_df):
        # test most methods for the class corresponding to each kind of feature
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("annotator", _feature)
            _explorer = _cls({"raw": example_raw_df})
            _explorer.plot()
            _ = _explorer.view()

    @staticmethod
    @pytest.mark.lite
    def test_labeling(example_raw_df):
        """
        No difference across text / image / audio.
        Pick the feature at random.
        """
        feature = random.choice(MAIN_FEATURES)
        explorer = get_explorer_class("annotator", feature)({"raw": example_raw_df})

        # empty click
        apply_event = ButtonClick(explorer.annotator_apply)
        explorer.annotator_apply._trigger_event(apply_event)

        # test standard selection
        for (_option_code, _selection, _expected_selected) in [
            (0, [0], [0]),
            # standard selection
            (0, [1], [1]),
            # set-union selection
            (1, [0, 2, 3], [0, 1, 2, 3]),
            # set-intersection selection
            (2, [0, 1, 2, 4], [0, 1, 2]),
            # set-difference selection
            (3, [2, 3], [0, 1]),
        ]:
            explorer.selection_option_box.active = _option_code
            explorer.sources["raw"].selected.indices = _selection
            explorer._store_selection()
            assert explorer.sources["raw"].selected.indices == _expected_selected

        # actual labeling
        assert (
            explorer.dfs["raw"].loc[[0, 1], "label"]
            == [module_config.ABSTAIN_DECODED] * 2
        ).all()
        explorer.annotator_input.value = "A"
        explorer.annotator_apply._trigger_event(apply_event)
        assert (explorer.dfs["raw"].loc[[0, 1], "label"] == ["A"] * 2).all()


@pytest.mark.core
class TestBokehSoftLabel:
    @staticmethod
    @pytest.mark.lite
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
    @pytest.mark.lite
    def test_filter_score(example_soft_label_df):
        """
        No difference across text / image / audio.
        Pick the feature at random.
        """
        feature = random.choice(MAIN_FEATURES)
        explorer = get_explorer_class("softlabel", feature)(
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
class TestBokehMargin:
    @staticmethod
    @pytest.mark.lite
    def test_init(example_margin_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("margin", _feature)
            _explorer = _cls({"raw": example_margin_df}, "label_1", "label_2")
            _explorer.plot("A")
            _explorer.plot("B")
            _ = _explorer.view()


@pytest.mark.core
class TestBokehSnorkel:
    @staticmethod
    @pytest.mark.lite
    def test_init(example_raw_df, example_labeled_df):
        for _feature in MAIN_FEATURES:
            _cls = get_explorer_class("snorkel", _feature)
            _explorer = _cls({"raw": example_raw_df, "labeled": example_labeled_df})
            _explorer.plot()
            _explorer.plot_lf(RANDOM_LABEL_LF, include=("C", "I", "M", "H"))
            _ = _explorer.view()

    @staticmethod
    @pytest.mark.lite
    def test_lf_labeling(example_raw_df, example_labeled_df):
        """
        No difference across text / image / audio.
        Just using text should suffice.
        """
        explorer = get_explorer_class("snorkel", "text")(
            {
                "raw": example_raw_df,
                "labeled": example_labeled_df,
            }
        )
        explorer.plot()
        initial_palette_size = len(explorer.palette)

        # create some dummy rules for predictable outcome
        lf_collection = subroutine_rules_from_text_df(explorer.dfs["raw"])
        narrow_rule_a = lf_collection["narrow_a"]
        narrow_rule_b = lf_collection["narrow_b"]
        broad_rule_a = lf_collection["broad_a"]
        broad_rule_b = lf_collection["broad_b"]
        # add a rule, check menu
        explorer.plot_lf(narrow_rule_b)
        assert explorer.lf_apply_trigger.menu == ["narrow_rule_b"]
        assert explorer.lf_filter_trigger.menu == ["narrow_rule_b"]

        # subscribe to a LF list, refresh, and check again
        lf_list = [narrow_rule_a, broad_rule_a]
        explorer.subscribed_lf_list = lf_list
        refresh_event = ButtonClick(explorer.lf_list_refresher)
        explorer.lf_list_refresher._trigger_event(refresh_event)
        lf_names_so_far = ["narrow_rule_a", "broad_rule_a"]
        assert explorer.lf_apply_trigger.menu == lf_names_so_far
        assert explorer.lf_filter_trigger.menu == lf_names_so_far

        # add an existing rule: menu, glyph, and view should stay the same
        old_narrow_a_lf = explorer.lf_data["narrow_rule_a"]["lf"]
        old_narrow_a_glyph_c = explorer.lf_data["narrow_rule_a"]["glyphs"]["C"]
        old_narrow_a_view_c = old_narrow_a_glyph_c.view
        explorer.plot_lf(narrow_rule_a)
        assert explorer.lf_apply_trigger.menu == lf_names_so_far
        assert explorer.lf_filter_trigger.menu == lf_names_so_far
        narrow_a_data_dict = explorer.lf_data["narrow_rule_a"]
        assert narrow_a_data_dict["lf"] is old_narrow_a_lf
        assert narrow_a_data_dict["glyphs"]["C"] is old_narrow_a_glyph_c
        assert narrow_a_data_dict["glyphs"]["C"].view is old_narrow_a_view_c

        # overwrite a rule: the dict reference in lf_data should stay the same
        # menu items and glyph references should stay the same
        # the view references of the glyphs should have changed
        narrow_rule_a = lf_collection["narrow_a_clone"]
        explorer.plot_lf(narrow_rule_a)
        assert explorer.lf_apply_trigger.menu == lf_names_so_far
        assert explorer.lf_filter_trigger.menu == lf_names_so_far
        assert narrow_a_data_dict["lf"] is not old_narrow_a_lf
        assert narrow_a_data_dict["glyphs"]["C"] is old_narrow_a_glyph_c
        assert narrow_a_data_dict["glyphs"]["C"].view is not old_narrow_a_view_c

        # empty click: nothing selected
        filter_event = MenuItemClick(explorer.lf_filter_trigger, item="broad_rule_a")
        apply_event = MenuItemClick(explorer.lf_apply_trigger, item="narrow_rule_a")
        explorer.lf_filter_trigger._trigger_event(filter_event)
        explorer.lf_apply_trigger._trigger_event(apply_event)

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
        explorer.lf_filter_trigger._trigger_event(filter_event)
        explorer.lf_apply_trigger._trigger_event(apply_event)

        first_six_labels = explorer.dfs["raw"]["label"].iloc[:6].tolist()
        assert first_six_labels == ["A"] * 6

        # add more rules, check menu again
        lf_list.append(narrow_rule_b)
        lf_list.append(broad_rule_b)
        explorer.lf_list_refresher._trigger_event(refresh_event)

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

        # use two pops to check against misremoval of renderers
        lf_list.pop()
        lf_list.pop()
        explorer.lf_list_refresher._trigger_event(refresh_event)
        assert len(explorer.palette) == initial_palette_size - len(lf_list)
