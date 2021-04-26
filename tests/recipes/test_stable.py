from hover import module_config
from hover.recipes.stable import _simple_annotator, _linked_annotator
from bokeh.events import ButtonClick, MenuItemClick

# from bokeh.document import Document


def test_simple_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _simple_annotator(dataset)

    annotator = objects["annotator"]

    initial_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}

    annotator.sources["raw"].selected.indices = [0, 1]
    annotator.annotator_input.value = "alt.atheism"

    # view selection
    view_event = ButtonClick(dataset.selection_viewer)
    dataset.selection_viewer._trigger_event(view_event)
    # dataset.sel_table.source.data is a {"field": []}-like dict
    assert len(dataset.sel_table.source.data["label"]) == 2

    # apply labels
    apply_event = ButtonClick(annotator.annotator_apply)
    annotator.annotator_apply._trigger_event(apply_event)
    labeled_slice = annotator.dfs["raw"][
        annotator.dfs["raw"]["label"] != module_config.ABSTAIN_DECODED
    ]
    assert labeled_slice.shape[0] == 2

    # commit changes to train set
    commit_event = MenuItemClick(dataset.data_committer, item="train")
    dataset.data_committer._trigger_event(commit_event)

    # deduplicate across dfs
    dedup_event = ButtonClick(dataset.dedup_trigger)
    dataset.dedup_trigger._trigger_event(dedup_event)
    new_sizes = {_key: _df.shape[0] for _key, _df in dataset.dfs.items()}
    assert new_sizes["raw"] == initial_sizes["raw"] - 2
    assert new_sizes["train"] == initial_sizes["train"] + 2

    # push dataset changes to annotator
    push_event = ButtonClick(dataset.update_pusher)
    dataset.update_pusher._trigger_event(push_event)


def test_linked_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _linked_annotator(dataset)

    # TODO: write more meaningful interactions and assertions
    assert objects["dataset"].dfs


# def test_servable_simple_annotator(mini_supervisable_text_dataset_embedded):
#    dataset = mini_supervisable_text_dataset_embedded.copy()
#    doc = Document()
#    handle = simple_annotator(dataset)
#    handle(doc)
