from hover.recipes import simple_annotator, linked_annotator
from bokeh.document import Document


def test_simple_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    doc = Document()
    handle = simple_annotator(dataset)
    handle(doc)


def test_linked_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    doc = Document()
    handle = linked_annotator(dataset)
    handle(doc)
