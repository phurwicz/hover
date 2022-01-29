from hover.recipes.stable import (
    _simple_annotator,
    _linked_annotator,
    simple_annotator,
    linked_annotator,
)
from bokeh.document import Document
from tests.recipes.local_helper import (
    action_view_selection,
    action_evict_selection,
    action_patch_selection,
    action_apply_labels,
    action_commit_selection,
    action_deduplicate,
    action_push_data,
)


def test_simple_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _simple_annotator(dataset)
    assert layout.visible

    annotator = objects["annotator"]
    initial_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}

    annotator.sources["raw"].selected.indices = [0, 1, 2]
    annotator.annotator_input.value = "alt.atheism"

    # view selected points in selection table
    view_data = action_view_selection(dataset)
    assert len(view_data["label"]) == 3

    # evict a point from selection
    dataset.sel_table.source.selected.indices = [1]
    old_view_data, new_view_data = action_evict_selection(dataset)
    assert len(old_view_data["label"]) == 3
    assert len(new_view_data["label"]) == 2

    # apply label(s)
    labeled_slice = action_apply_labels(annotator)
    assert labeled_slice.shape[0] == 2

    # commit and deduplicate
    action_commit_selection(dataset, subset="train")
    action_deduplicate(dataset)
    new_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}
    assert new_sizes["raw"] == initial_sizes["raw"] - 2
    assert new_sizes["train"] == initial_sizes["train"] + 2

    # refresh explorers
    action_push_data(dataset)

    # previous selections should be cleared after the push
    # prepare to patch a train-set label
    annotator.sources["train"].selected.indices = [0, 1]
    view_data = action_view_selection(dataset)
    assert len(view_data["label"]) == 2
    old_label = dataset.dfs["train"].at[0, "label"]
    new_label = "alt.atheism"
    if old_label == new_label:
        new_label = "rec.autos"
    dataset.sel_table.source.data["label"][0] = new_label
    dataset.sel_table.source.selected.indices = [0]
    assert dataset.dfs["train"].at[0, "label"] != new_label
    action_patch_selection(dataset)
    assert dataset.dfs["train"].at[0, "label"] == new_label


def test_linked_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _linked_annotator(dataset)
    assert layout.visible

    annotator, finder = objects["annotator"], objects["finder"]
    assert annotator.sources["raw"].selected.indices == []
    finder.sources["raw"].selected.indices = [0, 1, 2]
    assert annotator.sources["raw"].selected.indices == [0, 1, 2]


def test_servable_stable(mini_supervisable_text_dataset_embedded):
    for _recipe in [simple_annotator, linked_annotator]:
        dataset = mini_supervisable_text_dataset_embedded.copy()
        doc = Document()
        handle = _recipe(dataset)
        handle(doc)
