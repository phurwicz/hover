"""
Stable recipes whose function signatures should almost never change in the future.
"""
from bokeh.layouts import row, column
from hover.core.dataset import SupervisableDataset
from hover.core.explorer import BokehCorpusExplorer, BokehCorpusAnnotator
from hover.utils.bokeh_helper import servable


@servable(title="Simple Annotator")
def simple_annotator(dataset, height=600, width=600):
    """
    The most basic recipe, which nonetheless can be useful with decent 2-d embedding.

    Layout:

    sidebar | [annotate here]
    """
    # create explorers, setting up the first plots
    corpus_annotator = BokehCorpusAnnotator.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Annotator: apply labels to the selected points",
        height=height,
        width=width,
    )

    corpus_annotator.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(
        corpus_annotator, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_data_commit(corpus_annotator, {"raw": "raw"})

    sidebar = dataset.view()
    layout = row(sidebar, corpus_annotator.view())
    return layout


@servable(title="Linked Annotator")
def linked_annotator(dataset, height=600, width=600):
    """
    Leveraging CorpusExplorer which has the best search highlights.

    Layout:

    sidebar | [search here] | [annotate here]
    """
    # create explorers, setting up the first plots
    corpus_explorer = BokehCorpusExplorer.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Corpus: use the search widget for highlights",
        height=height,
        width=width,
    )

    corpus_annotator = BokehCorpusAnnotator.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Annotator: apply labels to the selected points",
        height=height,
        width=width,
    )

    corpus_explorer.plot()
    corpus_annotator.plot()

    # link coordinates and selections
    corpus_explorer.link_xy_range(corpus_annotator)
    corpus_explorer.link_selection("raw", corpus_annotator, "raw")

    # subscribe to dataset widgets
    dataset.subscribe_update_push(
        corpus_explorer, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_update_push(
        corpus_annotator, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_data_commit(corpus_annotator, {"raw": "raw"})

    sidebar = dataset.view()
    layout = row(sidebar, corpus_explorer.view(), corpus_annotator.view())
    return layout
