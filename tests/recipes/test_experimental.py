from hover.recipes.experimental import _active_learning, _snorkel_crosscheck

# from bokeh.document import Document
# from tests.recipes.local_helper import (
#    action_view_selection,
#    action_apply_labels,
#    action_commit_selection,
#    action_deduplicate,
#    action_push_data,
# )


def test_active_learning(
    mini_supervisable_text_dataset_embedded, dummy_vectorizer, dummy_vecnet_callback
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _active_learning(dataset, dummy_vectorizer, dummy_vecnet_callback)

    # TODO: add emulations of user activity
    assert objects


def test_snorkel_crosscheck(
    mini_supervisable_text_dataset_embedded, dummy_labeling_function_list
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _snorkel_crosscheck(dataset, dummy_labeling_function_list)
    # TODO: add emulations of user activity
    assert objects


# def test_active_learning(
#    mini_supervisable_text_dataset_embedded, dummy_vectorizer, dummy_vecnet_callback
# ):
#    dataset = mini_supervisable_text_dataset_embedded.copy()
#    doc = Document()
#    handle = active_learning(dataset, dummy_vectorizer, dummy_vecnet_callback)
#    handle(doc)
#
#
# def test_snorkel_crosscheck(
#    mini_supervisable_text_dataset_embedded, dummy_labeling_function_list
# ):
#    dataset = mini_supervisable_text_dataset_embedded.copy()
#    doc = Document()
#    handle = snorkel_crosscheck(dataset, dummy_labeling_function_list)
#    handle(doc)
