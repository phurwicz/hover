import numpy as np
from hover.recipes.experimental import _active_learning, _snorkel_crosscheck
from bokeh.events import ButtonClick, SelectionGeometry


def test_active_learning(
    mini_supervisable_text_dataset_embedded, dummy_vectorizer, dummy_vecnet_callback
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _active_learning(dataset, dummy_vectorizer, dummy_vecnet_callback)
    assert layout.visible

    initial_scores = dataset.dfs["raw"]["pred_score"].values.copy()

    finder, annotator = objects["finder"], objects["annotator"]
    softlabel = objects["softlabel"]
    coords_slider = softlabel._dynamic_widgets["patch_slider"]
    model_retrainer = objects["model_retrainer"]
    epochs_slider = objects["epochs_slider"]
    retrain_event = ButtonClick(model_retrainer)

    # train for 1 epoch
    epochs_slider.value = 1
    model_retrainer._trigger_event(retrain_event)
    first_scores = dataset.dfs["raw"]["pred_score"].values.copy()
    assert not np.allclose(first_scores, initial_scores)

    # emulating user interaction: slide coords to view manifold trajectory
    for _value in range(1, min(coords_slider.end + 1, 4)):
        coords_slider.value = _value

    # train for 1 more epoch
    model_retrainer._trigger_event(retrain_event)
    second_scores = dataset.dfs["raw"]["pred_score"].values
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


def test_snorkel_crosscheck(
    mini_supervisable_text_dataset_embedded, dummy_labeling_function_list
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _snorkel_crosscheck(dataset, dummy_labeling_function_list)
    assert layout.visible

    # TODO: add emulations of user activity
    assert objects
