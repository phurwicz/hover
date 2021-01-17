"""
???+ info "Stable recipes whose function signatures should almost never change in the future."
"""
from bokeh.layouts import row
from hover.utils.bokeh_helper import servable
from .subroutine import standard_annotator, standard_finder


@servable(title="Simple Annotator")
def simple_annotator(dataset, height=600, width=600):
    """
    ???+ info "Display the dataset with on a 2D map for annotation."


        | Arg       | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |

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
    ???+ info "Leveraging `BokehDataFinder` which has the best search highlights."

        | Arg       | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |

        Expected visual layout:

        | SupervisableDataset | BokehDataFinder    | BokehDataAnnotator |
        | :------------------ | :----------------- | :----------------- |
        | manage data subsets | search->highlight  | make annotations   |
    """
    finder = standard_finder(dataset, height=height, width=width)
    annotator = standard_annotator(dataset, height=height, width=width)

    # link coordinates and selections
    finder.link_xy_range(annotator)
    finder.link_selection("raw", annotator, "raw")

    sidebar = dataset.view()
    layout = row(sidebar, finder.view(), annotator.view())
    return layout
