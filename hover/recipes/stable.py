"""
???+ note "High-level functions to produce an interactive annotation interface."
    Stable recipes whose function signatures should almost never change in the future.
"""
from bokeh.layouts import row
from hover.utils.bokeh_helper import servable
from .subroutine import standard_annotator, standard_finder


@servable(title="Simple Annotator")
def simple_annotator(dataset, **kwargs):
    """
    ???+ note "Display the dataset with on a 2D map for annotation."

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` |       | kwargs to forward to each Bokeh figure |

        Expected visual layout:

        | SupervisableDataset | BokehDataAnnotator |
        | :------------------ | :----------------- |
        | manage data subsets | make annotations   |
    """
    layout, _ = _simple_annotator(dataset, **kwargs)
    return layout


def _simple_annotator(dataset, **kwargs):
    """
    ???+ note "Cousin of simple_annotator which exposes objects in the layout."
    """
    annotator = standard_annotator(dataset, **kwargs)

    sidebar = dataset.view()
    layout = row(sidebar, annotator.view())

    objects = {"dataset": dataset, "annotator": annotator, "sidebar": sidebar}
    return layout, objects


@servable(title="Linked Annotator")
def linked_annotator(dataset, **kwargs):
    """
    ???+ note "Display the dataset on a 2D map in two views, one for search and one for annotation."

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` |       | kwargs to forward to each Bokeh figure |

        Expected visual layout:

        | SupervisableDataset | BokehDataFinder     | BokehDataAnnotator |
        | :------------------ | :------------------ | :----------------- |
        | manage data subsets | search -> highlight | make annotations   |
    """
    layout, _ = _linked_annotator(dataset, **kwargs)
    return layout


def _linked_annotator(dataset, **kwargs):
    """
    ???+ note "Cousin of linked_annotator which exposes objects in the layout."
    """
    finder = standard_finder(dataset, **kwargs)
    annotator = standard_annotator(dataset, **kwargs)

    # link coordinates and selections
    finder.link_xy_range(annotator)
    for _key in ["raw", "train", "dev", "test"]:
        finder.link_selection(_key, annotator, _key)

    sidebar = dataset.view()
    layout = row(sidebar, finder.view(), annotator.view())

    objects = {
        "dataset": dataset,
        "annotator": annotator,
        "finder": finder,
        "sidebar": sidebar,
    }
    return layout, objects
