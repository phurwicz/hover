"""
???+ note "High-level functions to produce an interactive annotation interface."
    Experimental recipes whose function signatures might change significantly in the future. Use with caution.
"""
from bokeh.layouts import column
from .subroutine import (
    recipe_layout,
    standard_annotator,
    standard_finder,
    standard_snorkel,
    active_learning_components,
)
from hover.utils.bokeh_helper import servable


@servable(title="Snorkel Crosscheck")
def snorkel_crosscheck(dataset, lf_list, **kwargs):
    """
    ???+ note "Display the dataset for annotation, cross-checking with labeling functions."

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `lf_list` | `list`   | a list of callables decorated by `@hover.utils.snorkel_helper.labeling_function` |
        | `**kwargs` |       | kwargs to forward to each Bokeh figure |

        Expected visual layout:

        | SupervisableDataset | BokehSnorkelExplorer       | BokehDataAnnotator | BokehDataFinder     |
        | :------------------ | :------------------------- | :----------------- | :------------------ |
        | manage data subsets | inspect labeling functions | make annotations   | search and filter   |
    """
    layout, _ = _snorkel_crosscheck(dataset, lf_list, **kwargs)
    return layout


def _snorkel_crosscheck(dataset, lf_list, layout_style="horizontal", **kwargs):
    """
    ???+ note "Cousin of snorkel_crosscheck which exposes objects in the layout."
    """
    # building-block subroutines
    snorkel = standard_snorkel(dataset, **kwargs)
    snorkel.subscribed_lf_list = lf_list
    annotator = standard_annotator(dataset, **kwargs)
    finder = standard_finder(dataset, **kwargs)

    # plot labeling functions
    for _lf in lf_list:
        snorkel.plot_lf(_lf)
    snorkel.figure.legend.click_policy = "hide"

    # link selections
    annotator.link_selection(
        finder,
        {_key: _key for _key in ["raw", "train", "dev", "test"]},
    )
    # note that SnorkelExplorer has different subsets
    annotator.link_selection(
        snorkel,
        {"raw": "raw", "dev": "labeled"},
    )

    sidebar = dataset.view()
    layout = recipe_layout(
        sidebar, snorkel.view(), annotator.view(), finder.view(), style=layout_style
    )

    objects = {
        "dataset": dataset,
        "annotator": annotator,
        "finder": finder,
        "snorkel": snorkel,
        "sidebar": sidebar,
    }
    return layout, objects


@servable(title="Active Learning")
def active_learning(dataset, vecnet, **kwargs):
    """
    ???+ note "Display the dataset for annotation, putting a classification model in the loop."
        Currently works most smoothly with `VectorNet`.

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `vecnet`  | `VectorNet` | model to use in the loop          |
        | `**kwargs` |         | forwarded to each Bokeh figure       |

        Expected visual layout:

        | SupervisableDataset | BokehSoftLabelExplorer    | BokehDataAnnotator | BokehDataFinder     |
        | :------------------ | :------------------------ | :----------------- | :------------------ |
        | manage data subsets | inspect model predictions | make annotations   | search and filter   |
    """
    layout, _ = _active_learning(dataset, vecnet, **kwargs)
    return layout


def _active_learning(dataset, vecnet, layout_style="horizontal", **kwargs):
    """
    ???+ note "Cousin of active_learning which exposes objects in the layout."
    """

    # building-block subroutines
    annotator = standard_annotator(dataset, **kwargs)
    finder = standard_finder(dataset, **kwargs)
    softlabel, model_trainer = active_learning_components(dataset, vecnet, **kwargs)

    # link selections, noting that softlabel does not take "test"
    annotator.link_selection(
        finder,
        {_key: _key for _key in ["raw", "train", "dev", "test"]},
    )
    annotator.link_selection(
        softlabel,
        {_key: _key for _key in ["raw", "train", "dev"]},
    )

    sidebar = column(model_trainer, vecnet.view(), dataset.view())
    layout = recipe_layout(
        sidebar,
        *[_plot.view() for _plot in [softlabel, annotator, finder]],
        style=layout_style
    )

    objects = {
        "dataset": dataset,
        "annotator": annotator,
        "finder": finder,
        "sidebar": sidebar,
        "softlabel": softlabel,
        "model": vecnet,
        "model_trainer": model_trainer,
    }
    return layout, objects
