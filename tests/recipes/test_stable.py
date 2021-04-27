from hover import module_config
from hover.recipes.stable import _simple_annotator, _linked_annotator
from bokeh.events import ButtonClick, MenuItemClick

# import time
# from bokeh.document import Document


def action_view_selection(dataset):
    view_event = ButtonClick(dataset.selection_viewer)
    dataset.selection_viewer._trigger_event(view_event)
    # dataset.sel_table.source.data is a {"field": []}-like dict
    view_data = dataset.sel_table.source.data
    return view_data


def action_apply_labels(annotator):
    apply_event = ButtonClick(annotator.annotator_apply)
    annotator.annotator_apply._trigger_event(apply_event)
    labeled_slice = annotator.dfs["raw"][
        annotator.dfs["raw"]["label"] != module_config.ABSTAIN_DECODED
    ]
    return labeled_slice


def action_commit_selection(dataset):
    commit_event = MenuItemClick(dataset.data_committer, item="train")
    dataset.data_committer._trigger_event(commit_event)
    return dataset


def action_deduplicate(dataset):
    dedup_event = ButtonClick(dataset.dedup_trigger)
    dataset.dedup_trigger._trigger_event(dedup_event)
    return dataset


def action_push_data(dataset):
    push_event = ButtonClick(dataset.update_pusher)
    dataset.update_pusher._trigger_event(push_event)


def test_simple_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _simple_annotator(dataset)

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

    annotator, finder = objects["annotator"], objects["finder"]
    assert annotator.sources["raw"].selected.indices == []
    finder.sources["raw"].selected.indices = [0, 1, 2]

    # TODO: selected indices are synchronized through js_link
    # which seems to not propagate back to Python
    # time.sleep(1)
    # assert annotator.sources["raw"].selected.indices == [0, 1, 2]


# def test_servable_simple_annotator(mini_supervisable_text_dataset_embedded):
#    dataset = mini_supervisable_text_dataset_embedded.copy()
#    doc = Document()
#    handle = simple_annotator(dataset)
#    handle(doc)
