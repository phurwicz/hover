from hover import module_config
from hover.core.explorer.local_config import SEARCH_SCORE_FIELD
from hover.utils.snorkel_helper import labeling_function
from bokeh.events import SelectionGeometry
from tests.local_config import RANDOM_LABEL, PSEUDO_LABELS


MAIN_FEATURES = ["text", "image", "audio"]
FUNCTIONALITY_TO_SPECIAL_ARGS = {
    "annotator": tuple(),
    "finder": tuple(),
    "margin": ("label_1", "label_2"),
    "snorkel": tuple(),
    "softlabel": ("pred_label", "pred_score"),
}

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
        final=True,
    )
    return select_event


def almost_none_select(figure):
    select_event = SelectionGeometry(
        figure,
        geometry={
            "type": "poly",
            "x": [1e4 - 1e-6, 1e4 - 1e-6, 1e4 + 1e-6, 1e4 + 1e-6],
            "y": [1e4 - 1e-6, 1e4 + 1e-6, 1e4 + 1e-6, 1e4 - 1e-6],
            "sx": [None, None, None, None],
            "sy": [None, None, None, None],
        },
        final=True,
    )
    return select_event


def subroutine_search_source_response(explorer, search_callbacks):
    """
    Assumes search callbacks should be nontrivial, i.e. explorer sources are expected to change.
    """
    # initialize in-loop variable for comparison
    source = explorer.sources["raw"]
    _prev_scores = source.data[SEARCH_SCORE_FIELD].copy()
    for _callback in search_callbacks:
        _callback()
        _scores = source.data[SEARCH_SCORE_FIELD].copy()
        assert (
            _scores != _prev_scores
        ), f"Expected search scores to change, got {_prev_scores} (old) vs. {_scores} (new)"
        _prev_scores = _scores


def subroutine_selection_filter(explorer, filter_toggle, narrowing_callbacks):
    """
    Assumes narrowing callbacks to give strict nonempty subsets.
    """
    total_raw = explorer.dfs["raw"].shape[0]
    initial_select = list(range(total_raw))

    # emulate user interface: select everything through a SelectionGeometry event
    # note: bokeh's SelectionGeometry seems to not actually make a selection
    # note: the indices assignment has to happen before SelectionGeometry trigger
    # - this is for treating indices assignment as a manual select
    explorer.sources["raw"].selected.indices = initial_select[:]
    select_event = almost_global_select(explorer.figure)
    explorer.figure._trigger_event(select_event)
    assert explorer.sources["raw"].selected.indices == initial_select[:]

    # trigger the first callback without activating filter
    narrowing_callbacks[0]()
    assert explorer.sources["raw"].selected.indices == initial_select[:]

    # activate filter
    filter_toggle.active = [0]
    first_filter_select = explorer.sources["raw"].selected.indices[:]
    assert first_filter_select
    assert first_filter_select != initial_select
    assert set(first_filter_select).issubset(set(initial_select))

    # trigger the subsequent callbacks; selection should narrow
    _prev_select = first_filter_select[:]
    for _callback in narrowing_callbacks[1:]:
        _callback()
        _curr_select = explorer.sources["raw"].selected.indices[:]
        assert _curr_select
        assert _curr_select != _prev_select
        assert set(_curr_select).issubset(set(_prev_select))
        _prev_select = _curr_select[:]

    # deactivate filter
    filter_toggle.active = []
    unfilter_select = explorer.sources["raw"].selected.indices[:]
    assert unfilter_select == initial_select


def subroutine_rules_from_text_df(df):
    """
    Dummy rules for predictable outcome.
    """
    texts = df["text"].tolist()
    assert len(texts) >= 20, f"Expected at least 20 texts, got {len(texts)}"
    first_six_texts = set(texts[:6])
    first_ten_texts = set(texts[:10])

    def subroutine_lookup(query, pool, label):
        if query in pool:
            return label
        return module_config.ABSTAIN_DECODED

    @labeling_function(targets=["A"], name="narrow_rule_a")
    def narrow_rule_a_clone(row):
        return subroutine_lookup(row["text"], first_six_texts, "A")

    @labeling_function(targets=["A"])
    def narrow_rule_a(row):
        return subroutine_lookup(row["text"], first_six_texts, "A")

    @labeling_function(targets=["A"])
    def broad_rule_a(row):
        return subroutine_lookup(row["text"], first_ten_texts, "A")

    @labeling_function(targets=["B"])
    def narrow_rule_b(row):
        return subroutine_lookup(row["text"], first_six_texts, "B")

    @labeling_function(targets=["B"])
    def broad_rule_b(row):
        return subroutine_lookup(row["text"], first_ten_texts, "B")

    lf_collection = {
        "narrow_a_clone": narrow_rule_a_clone,
        "narrow_a": narrow_rule_a,
        "broad_a": broad_rule_a,
        "narrow_b": narrow_rule_b,
        "broad_b": broad_rule_b,
    }
    return lf_collection
