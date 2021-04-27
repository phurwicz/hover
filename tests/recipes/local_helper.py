from bokeh.events import ButtonClick, MenuItemClick
from hover import module_config


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
