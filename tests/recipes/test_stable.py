from hover.recipes.stable import _simple_annotator, _linked_annotator

# from bokeh.document import Document


def test_simple_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    layout, objects = _simple_annotator(dataset)

    # TODO: write more meaningful interactions and assertions
    assert objects["dataset"].dfs


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
