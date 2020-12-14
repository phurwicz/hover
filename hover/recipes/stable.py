"""
Stable recipes whose function signatures should almost never change in the future.
"""
from bokeh.layouts import row, column
from hover.core.dataset import SupervisableDataset
from hover.core.explorer import BokehCorpusExplorer, BokehCorpusAnnotator
from hover.utils.bokeh_helper import servable
from .subroutine import standard_annotator, standard_explorer


@servable(title="Simple Annotator")
def simple_annotator(dataset, height=600, width=600):
    """
    The most basic recipe, which nonetheless can be useful with decent 2-d embedding.

    Layout:

    sidebar | [annotate here]
    """
    annotator = standard_annotator(dataset, height, width)

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
    explorer = standard_explorer(dataset, height, width)
    annotator = standard_annotator(dataset, height, width)

    # link coordinates and selections
    explorer.link_xy_range(annotator)
    explorer.link_selection("raw", explorer, "raw")

    sidebar = dataset.view()
    layout = row(sidebar, corpus_explorer.view(), corpus_annotator.view())
    return layout
