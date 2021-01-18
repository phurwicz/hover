"""
???+ note "High-level functions to produce an interactive annotation interface."
    Stable recipes whose function signatures should almost never change in the future.
"""
from bokeh.layouts import row
from hover.utils.bokeh_helper import servable
from .subroutine import standard_annotator, standard_finder


@servable(title="Simple Annotator")
def simple_annotator(dataset, height=600, width=600):
    """
    ???+ note "Display the dataset with on a 2D map for annotation."

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `height`  | `int`    | height of each Bokeh explorer plot   |
        | `width`   | `int`    | width of each Bokeh explorer plot    |

        Expected visual layout:

        | SupervisableDataset | BokehDataAnnotator |
        | :------------------ | :----------------- |
        | manage data subsets | make annotations   |
    """
    annotator = standard_annotator(dataset, height=height, width=width)

    sidebar = dataset.view()
    layout = row(sidebar, annotator.view())
    return layout


@servable(title="Linked Annotator")
def linked_annotator(dataset, height=600, width=600):
    """
    ???+ note "Display the dataset on a 2D map in two views, one for search and one for annotation."

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `height`  | `int`    | height of each Bokeh explorer plot   |
        | `width`   | `int`    | width of each Bokeh explorer plot    |

        Expected visual layout:

        | SupervisableDataset | BokehDataFinder     | BokehDataAnnotator |
        | :------------------ | :------------------ | :----------------- |
        | manage data subsets | search -> highlight | make annotations   |
    """
    finder = standard_finder(dataset, height=height, width=width)
    annotator = standard_annotator(dataset, height=height, width=width)

    # link coordinates and selections
    finder.link_xy_range(annotator)
    finder.link_selection("raw", annotator, "raw")

    sidebar = dataset.view()
    layout = row(sidebar, finder.view(), annotator.view())
    return layout
