"""
???+ note "High-level functions to produce an interactive annotation interface."
    Experimental recipes whose function signatures might change significantly in the future. Use with caution.
"""
from bokeh.models import Button
from bokeh.layouts import column
from .subroutine import (
    recipe_layout,
    standard_annotator,
    standard_finder,
    standard_snorkel,
    standard_softlabel,
)
from hover.utils.bokeh_helper import servable
from rich.console import Console
import numpy as np


@servable(title="Snorkel Crosscheck")
def snorkel_crosscheck(dataset, lf_list, **kwargs):
    """
    ???+ note "Display the dataset for annotation, cross-checking with labeling functions."
        Use the train set to check labeling functions; use the labeling functions to hint at potential annotation.

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `lf_list` | `list`   | a list of callables decorated by `@hover.utils.snorkel_helper.labeling_function` |
        | `**kwargs` |       | kwargs to forward to each Bokeh figure |

        Expected visual layout:

        | SupervisableDataset | BokehSnorkelExplorer       | BokehDataAnnotator |
        | :------------------ | :------------------------- | :----------------- |
        | manage data subsets | inspect labeling functions | make annotations   |
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

    # plot labeling functions
    for _lf in lf_list:
        snorkel.plot_lf(_lf)
    snorkel.figure.legend.click_policy = "hide"

    # link coordinates and selections
    snorkel.link_xy_range(annotator)
    snorkel.link_selection_options(annotator)
    snorkel.link_selection("raw", annotator, "raw")
    snorkel.link_selection("labeled", annotator, "train")

    sidebar = dataset.view()
    layout = recipe_layout(
        sidebar, snorkel.view(), annotator.view(), style=layout_style
    )

    objects = {
        "dataset": dataset,
        "annotator": annotator,
        "snorkel": snorkel,
        "sidebar": sidebar,
    }
    return layout, objects


@servable(title="Active Learning")
def active_learning(dataset, vecnet_callback, **kwargs):
    """
    ???+ note "Display the dataset for annotation, putting a classification model in the loop."
        Currently works most smoothly with `VectorNet`.

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `vecnet_callback` | `callable` | function that creates a `VecNet` based on a `SupervisableDataset`|
        | `**kwargs` |         | forwarded to each Bokeh figure       |

        Expected visual layout:

        | SupervisableDataset | BokehSoftLabelExplorer    | BokehDataAnnotator | BokehDataFinder     |
        | :------------------ | :------------------------ | :----------------- | :------------------ |
        | manage data subsets | inspect model predictions | make annotations   | search -> highlight |
    """
    layout, _ = _active_learning(dataset, vecnet_callback, **kwargs)
    return layout


def _active_learning(dataset, vecnet_callback, layout_style="horizontal", **kwargs):
    """
    ???+ note "Cousin of active_learning which exposes objects in the layout."
    """
    console = Console()
    feature_key = dataset.__class__.FEATURE_KEY

    # building-block subroutines
    softlabel = standard_softlabel(dataset, **kwargs)
    annotator = standard_annotator(dataset, **kwargs)
    finder = standard_finder(dataset, **kwargs)

    # link coordinates, omitting the softlabel
    finder.link_xy_range(annotator)

    # link selections, noting that softlabel does not take "test"
    finder.link_selection_options(annotator)
    finder.link_selection_options(softlabel)
    for _key in ["raw", "train", "dev"]:
        softlabel.link_selection(_key, annotator, _key)
        softlabel.link_selection(_key, finder, _key)
    finder.link_selection("test", annotator, "test")

    # patch coordinates for representational similarity analysis
    softlabel.value_patch("x", "x_traj", title="Manifold trajectory step")
    softlabel.value_patch("y", "y_traj")

    # recipe-specific widget
    model = vecnet_callback(dataset)
    model_trainer = Button(label="Train model", button_type="primary")

    def retrain_model():
        """
        Callback subfunction 1 of 2.
        """
        model_trainer.disabled = True
        console.print("Start training... button will be disabled temporarily.")
        dataset.setup_label_coding()

        train_loader = model.prepare_loader(dataset, "train", smoothing_coeff=0.2)
        if dataset.dfs["dev"].shape[0] > 0:
            dev_loader = model.prepare_loader(dataset, "dev")
        else:
            dataset._warn("dev set is empty, borrowing train set for validation.")
            dev_loader = train_loader

        _ = model.train(train_loader, dev_loader)
        model.save()
        console.print("-- 1/2: retrained model")

    def update_softlabel_plot():
        """
        Callback subfunction 2 of 2.
        """
        # combine inputs and compute outputs of all non-test subsets
        use_subsets = ("raw", "train", "dev")
        inps = []
        for _key in use_subsets:
            inps.extend(dataset.dfs[_key][feature_key].tolist())

        probs = model.predict_proba(inps)
        labels = [dataset.label_decoder[_val] for _val in probs.argmax(axis=-1)]
        scores = probs.max(axis=-1).tolist()
        traj_arr, seq_arr, disparity_arr = model.manifold_trajectory(
            inps,
            points_per_step=5,
        )

        offset = 0
        for _key in use_subsets:
            _length = dataset.dfs[_key].shape[0]
            # skip subset if empty
            if _length > 0:
                _slice = slice(offset, offset + _length)
                dataset.dfs[_key]["pred_label"] = labels[_slice]
                dataset.dfs[_key]["pred_score"] = scores[_slice]
                # for each dimension: all steps, selected slice
                _x_traj = traj_arr[:, _slice, 0]
                _y_traj = traj_arr[:, _slice, 1]
                # for each dimension: selected slice, all steps
                _x_traj = list(np.swapaxes(_x_traj, 0, 1))
                _y_traj = list(np.swapaxes(_y_traj, 0, 1))
                dataset.dfs[_key]["x_traj"] = _x_traj
                dataset.dfs[_key]["y_traj"] = _y_traj

                offset += _length

        softlabel._dynamic_callbacks["adjust_patch_slider"]()
        softlabel._update_sources()
        model_trainer.disabled = False
        console.print("-- 2/2: updated predictions. Training button is re-enabled.")

    def callback_sequence():
        """
        Overall callback function.
        """
        retrain_model()
        update_softlabel_plot()

    model_trainer.on_click(callback_sequence)
    sidebar = column(model_trainer, model.view(), dataset.view())
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
        "model": model,
        "model_trainer": model_trainer,
    }
    return layout, objects
