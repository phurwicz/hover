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
    layout, objects = _simple_annotator(dataset, layout_style="horizontal")
    assert layout.visible

    annotator = objects["annotator"]
    initial_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}

    raw_view_select = [0, 1, 2, 3, 6, 10]
    annotator.sources["raw"].selected.indices = raw_view_select[:]
    annotator.annotator_input.value = "alt.atheism"

    # view selected points in selection table
    view_data = action_view_selection(dataset)
    assert len(view_data["label"]) == len(raw_view_select)

    # evict a point from selection
    evict_idx = 5
    # prepare expected texts post eviction
    expected_texts = dataset.dfs["raw"].loc[raw_view_select, "text"].tolist()
    expected_texts.pop(evict_idx)
    # execute eviction
    dataset.sel_table.source.selected.indices = [evict_idx]
    old_view_data, new_view_data = action_evict_selection(dataset)
    # check the number of points before/after eviction
    assert len(old_view_data["label"]) == len(raw_view_select)
    assert len(new_view_data["label"]) == len(raw_view_select) - 1
    # check specific feature values
    assert new_view_data["text"] == expected_texts

    # apply label(s)
    labeled_slice = action_apply_labels(annotator)
    expected_labeled_count = len(raw_view_select) - 1
    assert labeled_slice.shape[0] == expected_labeled_count

    # commit and deduplicate
    action_commit_selection(dataset, subset="train")
    action_deduplicate(dataset)
    new_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}
    assert new_sizes["raw"] == initial_sizes["raw"] - expected_labeled_count
    assert new_sizes["train"] == initial_sizes["train"] + expected_labeled_count

    # refresh explorers
    action_push_data(dataset)

    # previous selections should be cleared after the push
    # prepare to patch a train-set label
    assert annotator.sources["raw"].selected.indices == []
    assert annotator.sources["train"].selected.indices == []
    raw_view_select, train_view_select = [0, 1, 2], [0, 2, 4]
    annotator.sources["raw"].selected.indices = raw_view_select[:]
    annotator.sources["train"].selected.indices = train_view_select[:]
    view_data = action_view_selection(dataset)
    assert len(view_data["label"]) == len(raw_view_select + train_view_select)

    # check feature-based lookup
    raw_idx_to_patch = 5
    text_to_patch = dataset.sel_table.source.data["text"][raw_idx_to_patch]
    subset_to_patch, idx_to_patch = dataset.locate_by_feature_value(text_to_patch)
    assert subset_to_patch == "train"
    assert idx_to_patch == (raw_view_select + train_view_select)[raw_idx_to_patch]
    # prepare an edit patch
    old_label = dataset.dfs[subset_to_patch].at[idx_to_patch, "label"]
    new_label = "alt.atheism"
    if old_label == new_label:
        new_label = "rec.autos"
    dataset.sel_table.source.data["label"][raw_idx_to_patch] = new_label
    dataset.sel_table.source.selected.indices = [raw_idx_to_patch]
    # execute patch
    assert dataset.dfs[subset_to_patch].at[idx_to_patch, "label"] != new_label
    action_patch_selection(dataset)
    assert dataset.dfs[subset_to_patch].at[idx_to_patch, "label"] == new_label


def test_linked_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _linked_annotator(dataset, layout_style="vertical")
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
