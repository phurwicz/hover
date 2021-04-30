"""
???+ note "High-level functions to produce an interactive annotation interface."
    Experimental recipes whose function signatures might change significantly in the future. Use with caution.
"""
from bokeh.layouts import row, column
from bokeh.models import Button, Slider
from .subroutine import (
    standard_annotator,
    standard_finder,
    standard_snorkel,
    standard_softlabel,
)
from hover.utils.bokeh_helper import servable
from wasabi import msg as logger
import numpy as np


@servable(title="Snorkel Crosscheck")
def snorkel_crosscheck(dataset, lf_list, **kwargs):
    """
    ???+ note "Display the dataset for annotation, cross-checking with labeling functions."
        Use the dev set to check labeling functions; use the labeling functions to hint at potential annotation.

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


def _snorkel_crosscheck(dataset, lf_list, **kwargs):
    """
    ???+ note "Cousin of snorkel_crosscheck which exposes objects in the layout."
    """
    # building-block subroutines
    snorkel = standard_snorkel(dataset, **kwargs)
    annotator = standard_annotator(dataset, **kwargs)

    # plot labeling functions
    for _lf in lf_list:
        snorkel.plot_lf(_lf)
    snorkel.figure.legend.click_policy = "hide"

    # link coordinates and selections
    snorkel.link_xy_range(annotator)
    snorkel.link_selection("raw", annotator, "raw")
    snorkel.link_selection("labeled", annotator, "dev")

    sidebar = dataset.view()
    layout = row(sidebar, snorkel.view(), annotator.view())

    objects = {
        "dataset": dataset,
        "annotator": annotator,
        "snorkel": snorkel,
        "sidebar": sidebar,
    }
    return layout, objects


@servable(title="Active Learning")
def active_learning(dataset, vectorizer, vecnet_callback, **kwargs):
    """
    ???+ note "Display the dataset for annotation, putting a classification model in the loop."
        Currently works most smoothly with `VectorNet`.

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `vectorizer` | `callable` | the feature -> vector function  |
        | `vecnet_callback` | `callable` | the (dataset, vectorizer) -> `VecNet` function|
        | `**kwargs` |       | kwargs to forward to each Bokeh figure |

        Expected visual layout:

        | SupervisableDataset | BokehSoftLabelExplorer    | BokehDataAnnotator | BokehDataFinder     |
        | :------------------ | :------------------------ | :----------------- | :------------------ |
        | manage data subsets | inspect model predictions | make annotations   | search -> highlight |
    """
    layout, _ = _active_learning(dataset, vectorizer, vecnet_callback, **kwargs)
    return layout


def _active_learning(dataset, vectorizer, vecnet_callback, **kwargs):
    """
    ???+ note "Cousin of active_learning which exposes objects in the layout."
    """
    # building-block subroutines
    softlabel = standard_softlabel(dataset, **kwargs)
    annotator = standard_annotator(dataset, **kwargs)
    finder = standard_finder(dataset, **kwargs)

    # link coordinates, omitting the softlabel
    finder.link_xy_range(annotator)

    # link selections, noting that softlabel does not take "test"
    for _key in ["raw", "train", "dev"]:
        softlabel.link_selection(_key, annotator, _key)
        softlabel.link_selection(_key, finder, _key)
    finder.link_selection("test", annotator, "test")

    # patch coordinates for representational similarity analysis
    softlabel.value_patch("x", "x_traj", title="Manifold trajectory step")
    softlabel.value_patch("y", "y_traj")

    # recipe-specific widget
    def setup_model_retrainer():
        model_retrainer = Button(label="Train model", button_type="primary")
        epochs_slider = Slider(start=1, end=20, value=1, step=1, title="# epochs")

        def retrain_model():
            """
            Callback function.
            """
            model_retrainer.disabled = True
            logger.info("Start training... button will be disabled temporarily.")
            dataset.setup_label_coding()
            model = vecnet_callback(dataset, vectorizer)

            train_loader = dataset.loader("train", vectorizer, smoothing_coeff=0.2)
            if dataset.dfs["dev"].shape[0] > 0:
                dev_loader = dataset.loader("dev", vectorizer)
            else:
                dataset._warn("Dev set is empty, borrowing train set for validation.")
                dev_loader = train_loader

            _ = model.train(train_loader, dev_loader, epochs=epochs_slider.value)
            model.save()
            logger.good("-- 1/2: retrained model")

            # combine inputs and compute outputs of all non-test subsets
            use_subsets = ("raw", "train", "dev")
            inps = []
            for _key in use_subsets:
                inps.extend(dataset.dfs[_key]["text"].tolist())

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
            model_retrainer.disabled = False
            logger.good("-- 2/2: updated predictions. Training button is re-enabled.")

        model_retrainer.on_click(retrain_model)
        return model_retrainer, epochs_slider

    model_retrainer, epochs_slider = setup_model_retrainer()
    sidebar = column(row(model_retrainer, epochs_slider), dataset.view())
    layout = row(sidebar, *[_plot.view() for _plot in [softlabel, annotator, finder]])

    objects = {
        "dataset": dataset,
        "annotator": annotator,
        "finder": finder,
        "sidebar": sidebar,
        "softlabel": softlabel,
        "model_retrainer": model_retrainer,
        "epochs_slider": epochs_slider,
    }
    return layout, objects
