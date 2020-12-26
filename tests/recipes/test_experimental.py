from hover.recipes.experimental import active_learning, snorkel_crosscheck
from bokeh.document import Document


def test_active_learning(
    mini_supervisable_text_dataset_embedded, dummy_vectorizer, dummy_vecnet_callback
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    doc = Document()
    handle = active_learning(dataset, dummy_vectorizer, dummy_vecnet_callback)
    handle(doc)


def test_snorkel_crosscheck(
    mini_supervisable_text_dataset_embedded, dummy_labeling_function_list
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    doc = Document()
    handle = snorkel_crosscheck(dataset, dummy_labeling_function_list)
    handle(doc)
