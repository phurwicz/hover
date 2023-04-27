import pytest
import numpy as np
from hover.recipes.experimental import (
    _active_learning,
    _snorkel_crosscheck,
    active_learning,
    snorkel_crosscheck,
)
from hover.module_config import DataFrame as DF
from bokeh.events import ButtonClick, SelectionGeometry
from .local_helper import execute_handle_function


def test_active_learning(example_text_dataset, dummy_vecnet_callback):
    def read_scores(dataset, subset):
        return DF.series_values(dataset.dfs[subset]["pred_score"]).copy()

    dataset = example_text_dataset.copy()
    vecnet = dummy_vecnet_callback(dataset)
    layout, objects = _active_learning(dataset, vecnet)
    assert layout.visible

    initial_scores = read_scores(dataset, "raw")

    finder, annotator = objects["finder"], objects["annotator"]
    softlabel = objects["softlabel"]
    coords_slider = softlabel._dynamic_widgets["patch_slider"]
    model_trainer = objects["model_trainer"]
    train_event = ButtonClick(model_trainer)

    # train for default number of epochs
    model_trainer._trigger_event(train_event)
    first_scores = read_scores(dataset, "raw")
    assert not np.allclose(first_scores, initial_scores)

    # emulating user interaction: slide coords to view manifold trajectory
    for _value in range(1, min(coords_slider.end + 1, 4)):
        coords_slider.value = _value

    # train for 1 more epoch
    model_trainer._trigger_event(train_event)
    second_scores = read_scores(dataset, "raw")
    assert not np.allclose(second_scores, first_scores)
    # take 25 and 75 percentiles of scores for later use
    range_low, range_high = np.percentile(second_scores, [25, 75]).tolist()

    # emulate user interface: select everything through a SelectionGeometry event
    total_raw = softlabel.dfs["raw"].shape[0]
    initial_select = list(range(total_raw))
    # check linked selection
    assert annotator.sources["raw"].selected.indices == []
    softlabel.sources["raw"].selected.indices = initial_select
    box_select = SelectionGeometry(
        softlabel.figure,
        geometry={
            "type": "poly",
            "sx": [-1e4, -1e4, 1e4, 1e4],
            "sy": [-1e4, 1e4, 1e4, -1e4],
            "x": [None, None, None, None],
            "y": [None, None, None, None],
        },
    )
    softlabel.figure._trigger_event(box_select)
    assert annotator.sources["raw"].selected.indices == initial_select

    # check score filtering
    # nothing happens when filter is inactive
    softlabel.score_range.value = (range_low, range_high)
    assert softlabel.sources["raw"].selected.indices == initial_select
    # activate score filter
    softlabel.score_filter_box.active = [0]
    first_select = softlabel.sources["raw"].selected.indices[:]
    assert first_select != initial_select
    assert set(first_select).issubset(set(initial_select))
    assert first_select == annotator.sources["raw"].selected.indices

    # check regex co-filtering
    finder.search_filter_box.active = [0]
    finder.search_pos.value = r"(?i)s[aeiou]\ "
    second_select = softlabel.sources["raw"].selected.indices[:]
    assert second_select != first_select
    assert set(second_select).issubset(set(first_select))

    # check filter interaction: untoggle score filter
    softlabel.score_filter_box.active = []
    third_select = softlabel.sources["raw"].selected.indices[:]
    assert third_select != second_select
    assert set(second_select).issubset(set(third_select))

    # deactivate regex filter too
    finder.search_filter_box.active = []
    unfilter_select = softlabel.sources["raw"].selected.indices[:]
    assert unfilter_select == initial_select


def test_snorkel_crosscheck(example_audio_dataset, dummy_labeling_function_list):
    dataset = example_audio_dataset.copy()
    layout, objects = _snorkel_crosscheck(dataset, dummy_labeling_function_list)
    assert layout.visible

    # TODO: add emulations of user activity
    assert objects


@pytest.mark.lite
def test_servable_experimental(
    example_text_dataset,
    dummy_vecnet_callback,
    dummy_labeling_function_list,
):
    # one dataset for each recipe
    dataset = example_text_dataset.copy()
    vecnet = dummy_vecnet_callback(dataset)
    active = active_learning(dataset, vecnet)

    dataset = example_text_dataset.copy()
    snorkel = snorkel_crosscheck(dataset, dummy_labeling_function_list)

    for handle in [active, snorkel]:
        execute_handle_function(handle)
