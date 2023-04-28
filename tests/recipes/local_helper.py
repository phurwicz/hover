import time
import operator
from bokeh.document import Document
from bokeh.events import ButtonClick, MenuItemClick
from hover import module_config


def action_view_selection(dataset):
    view_event = ButtonClick(dataset.selection_viewer)
    dataset.selection_viewer._trigger_event(view_event)
    # dataset.sel_table.source.data is a {"field": []}-like dict
    view_data = dataset.sel_table.source.data.copy()
    return view_data


def action_evict_selection(dataset):
    old_view_data = dataset.sel_table.source.data.copy()
    evict_event = ButtonClick(dataset.selection_evictor)
    dataset.selection_evictor._trigger_event(evict_event)
    new_view_data = dataset.sel_table.source.data.copy()
    return old_view_data, new_view_data


def action_patch_selection(dataset):
    patch_event = ButtonClick(dataset.selection_patcher)
    dataset.selection_patcher._trigger_event(patch_event)


def action_apply_labels(annotator):
    apply_event = ButtonClick(annotator.annotator_apply)
    annotator.annotator_apply._trigger_event(apply_event)
    labeled_slice = annotator.dfs["raw"].filter_rows_by_operator(
        "label", operator.ne, module_config.ABSTAIN_DECODED
    )()
    return labeled_slice


def action_commit_selection(dataset, subset="train"):
    commit_event = MenuItemClick(dataset.data_committer, item=subset)
    dataset.data_committer._trigger_event(commit_event)


def action_deduplicate(dataset):
    dedup_event = ButtonClick(dataset.dedup_trigger)
    dataset.dedup_trigger._trigger_event(dedup_event)


def action_push_data(dataset):
    push_event = ButtonClick(dataset.update_pusher)
    dataset.update_pusher._trigger_event(push_event)


def execute_handle_function(handle):
    doc = Document()
    handle(doc)
    # a few seconds to activate timed callcacks
    time.sleep(10)
    for wrapped_callback in doc.session_callbacks:
        wrapped_callback.callback()
