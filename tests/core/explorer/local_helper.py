from hover.utils.snorkel_helper import labeling_function
from bokeh.events import SelectionGeometry
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
    )
    return select_event


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
