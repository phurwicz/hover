from hover.recipes.stable import (
    _simple_annotator,
    _linked_annotator,
    simple_annotator,
    linked_annotator,
)
from bokeh.document import Document
from tests.recipes.local_helper import (
    action_view_selection,
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

    annotator.sources["raw"].selected.indices = [0, 1]
    annotator.annotator_input.value = "alt.atheism"

    view_data = action_view_selection(dataset)
    assert len(view_data["label"]) == 2

    labeled_slice = action_apply_labels(annotator)
    assert labeled_slice.shape[0] == 2

    action_commit_selection(dataset)
    action_deduplicate(dataset)
    new_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}
    assert new_sizes["raw"] == initial_sizes["raw"] - 2
    assert new_sizes["train"] == initial_sizes["train"] + 2

    action_push_data(dataset)


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
