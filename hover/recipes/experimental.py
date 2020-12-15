"""
Experimental recipes whose function signatures might change significantly in the future. Use with caution.
"""
from bokeh.layouts import row, column
from bokeh.models import Button, Slider
from .subroutine import (
    standard_annotator,
    standard_explorer,
    standard_snorkel,
    standard_softlabel,
)
from hover.utils.bokeh_helper import servable
from wasabi import msg as logger
import pandas as pd


@servable(title="Snorkel Crosscheck")
def snorkel_crosscheck(dataset, lf_list, height=600, width=600):
    """
    Use the dev set to check labeling functions; use the labeling functions to hint at potential annotation.

    Layout:

    sidebar | [inspect LFs here] | [annotate here]
    """
    # building-block subroutines
    snorkel = standard_snorkel(dataset, height=height, width=width)
    annotator = standard_annotator(dataset, height=height, width=width)

    # plot labeling functions
    for _lf in lf_list:
        snorkel.plot_lf(_lf)
    snorkel.figure.legend.click_policy = "hide"

    # link coordinates and selections
    snorkel.link_xy_range(annotator)
    snorkel.link_selection("raw", annotator, "raw")

    sidebar = dataset.view()
    layout = row(sidebar, snorkel.view(), annotator.view())
    return layout


@servable(title="Active Learning")
def active_learning(dataset, vectorizer, vecnet_callback, height=600, width=600):
    """
    Place a VectorNet in the loop.

    Layout:

    sidebar | [inspect soft labels here] | [annotate here] | [search here]
    """
    # building-block subroutines
    softlabel = standard_softlabel(dataset, height=height, width=width)
    annotator = standard_annotator(dataset, height=height, width=width)
    explorer = standard_explorer(dataset, height=height, width=width)

    # link coordinates and selections
    softlabel.link_xy_range(annotator)
    softlabel.link_xy_range(explorer)
    softlabel.link_selection("raw", annotator, "raw")
    softlabel.link_selection("raw", explorer, "raw")

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
            dev_loader = dataset.loader("dev", vectorizer)

            _ = model.train(train_loader, dev_loader, epochs=epochs_slider.value)
            logger.good("-- 1/2: retrained model")

            for _key in ["raw", "train", "dev"]:
                _probs = model.predict_proba(dataset.dfs[_key]["text"].tolist())
                _labels = [
                    dataset.label_decoder[_val] for _val in _probs.argmax(axis=-1)
                ]
                _scores = _probs.max(axis=-1).tolist()
                dataset.dfs[_key]["pred_label"] = pd.Series(_labels)
                dataset.dfs[_key]["pred_score"] = pd.Series(_scores)

            softlabel._update_sources()
            softlabel.plot()
            model_retrainer.disabled = False
            logger.good("-- 2/2: updated predictions. Training button is re-enabled.")

        model_retrainer.on_click(retrain_model)
        return model_retrainer, epochs_slider

    model_retrainer, epochs_slider = setup_model_retrainer()
    sidebar = column(model_retrainer, epochs_slider, dataset.view())
    layout = row(sidebar, *[_plot.view() for _plot in [softlabel, annotator, explorer]])
    return layout
