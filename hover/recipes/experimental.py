"""
Experimental recipes whose function signatures might change significantly in the future. Use with caution.
"""
from bokeh.layouts import row, column
from bokeh.models import Button, Slider
from hover.core.dataset import SupervisableDataset
from hover.core.explorer import (
    BokehCorpusExplorer,
    BokehCorpusAnnotator,
    BokehSoftLabelExplorer,
    BokehSnorkelExplorer,
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
    # create explorers, setting up the first plots
    snorkel_explorer = BokehSnorkelExplorer.from_dataset(
        dataset,
        {"raw": "raw", "dev": "labeled"},
        title="Snorkel: square for correct, x for incorrect, + for missed, o for hit; click on legends to hide or show LF",
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

    snorkel_explorer.plot()
    for _lf in lf_list:
        snorkel_explorer.plot_lf(_lf)
    snorkel_explorer.figure.legend.click_policy = "hide"
    corpus_annotator.plot()

    # link coordinates and selections
    snorkel_explorer.link_xy_range(corpus_annotator)
    snorkel_explorer.link_selection("raw", corpus_annotator, "raw")

    # subscribe to dataset widgets
    dataset.subscribe_update_push(snorkel_explorer, {"raw": "raw", "dev": "labeled"})
    dataset.subscribe_update_push(
        corpus_annotator, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_data_commit(corpus_annotator, {"raw": "raw"})

    sidebar = dataset.view()
    layout = row(sidebar, snorkel_explorer.view(), corpus_annotator.view())
    return layout


@servable(title="Active Learning")
def active_learning(dataset, vectorizer, vecnet_callback, height=600, width=600):
    """
    Place a VectorNet in the loop.

    Layout:

    sidebar | [inspect soft labels here] | [annotate here] | [search here]
    """
    # create explorers, setting up the first plots
    # link coordinates and selections
    # subscribe to dataset widgets
    softlabel_explorer = BokehSoftLabelExplorer.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev"]},
        "pred_label",
        "pred_score",
        title="Prediction Visualizer: retrain model and locate confusions",
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

    corpus_explorer = BokehCorpusExplorer.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Corpus: use the search widget for highlights",
        height=height,
        width=width,
    )

    softlabel_explorer.plot()
    corpus_annotator.plot()
    corpus_explorer.plot()

    # link coordinates and selections
    softlabel_explorer.link_xy_range(corpus_annotator)
    softlabel_explorer.link_xy_range(corpus_explorer)
    softlabel_explorer.link_selection("raw", corpus_annotator, "raw")
    softlabel_explorer.link_selection("raw", corpus_explorer, "raw")

    # subscribe to dataset widgets
    dataset.subscribe_update_push(
        softlabel_explorer, {_k: _k for _k in ["raw", "train", "dev"]}
    )
    dataset.subscribe_update_push(
        corpus_annotator, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_update_push(
        corpus_explorer, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_data_commit(corpus_annotator, {"raw": "raw"})

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
            model = vecnet_callback()

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

            softlabel_explorer._update_sources()
            softlabel_explorer.plot()
            model_retrainer.disabled = False
            logger.good("-- 2/2: updated predictions. Training button is re-enabled.")

        model_retrainer.on_click(retrain_model)
        return model_retrainer, epochs_slider

    model_retrainer, epochs_slider = setup_model_retrainer()
    sidebar = column(model_retrainer, epochs_slider, dataset.view())
    layout = row(
        sidebar,
        *[
            _plot.view()
            for _plot in [softlabel_explorer, corpus_annotator, corpus_explorer]
        ]
    )
    return layout
