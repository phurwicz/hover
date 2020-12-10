"""
Experimental recipes whose function signatures might change significantly in the future. Use with caution.
"""
from bokeh.layouts import row, column
from hover.core.dataset import SupervisableDataset
from hover.core.explorer import (
    BokehForLabeledText,
    BokehCorpusExplorer,
    BokehCorpusAnnotator,
    BokehSoftLabelExplorer,
    BokehSnorkelExplorer,
)
from hover.utils.bokeh_helper import servable


def dataset_sync_explorer(dataset, explorer, subset_mapping):
    """
    Subscribe the sources of an explorer to a SupervisableDataset.
    """
    assert isinstance(dataset, SupervisableDataset)
    assert isinstance(explorer, BokehForLabeledText)

    def push_to_explorer():
        df_dict = {_v: dataset.dfs[_k] for _k, _v in subset_mapping.items()}
        explorer._setup_dfs(df_dict)
        explorer._update_sources()

    dataset.subscribe_update_pusher(push_to_explorer)


@servable(title="Simple Annotator")
def simple_annotator(dataset, height=600, width=600):
    """
    The most basic recipe, which nonetheless can be useful with decent 2-d embedding.

    Layout:

    sidebar | [annotate here]
    """
    corpus_annotator = BokehCorpusAnnotator(
        {"raw": dataset.dfs["raw"]},
        title="Annotator: apply labels to the selected points",
        height=height,
        width=width,
    )

    corpus_annotator.plot()
    dataset_sync_explorer(dataset, corpus_annotator, {"raw": "raw"})

    sidebar = column(dataset.update_pusher, dataset.pop_table)
    layout = row(sidebar, corpus_annotator.view())
    return layout


@servable(title="Linked Annotator")
def linked_annotator(dataset, height=600, width=600):
    """
    Leveraging CorpusExplorer which has the best search highlights.

    Layout:

    sidebar | [search here] | [annotate here]
    """
    corpus_explorer = BokehCorpusExplorer(
        {"raw": dataset.dfs["raw"]},
        title="Corpus: use the search widget for highlights",
        height=height,
        width=width,
    )

    corpus_annotator = BokehCorpusAnnotator(
        {"raw": dataset.dfs["raw"]},
        title="Annotator: apply labels to the selected points",
        height=height,
        width=width,
    )

    corpus_explorer.plot()
    corpus_annotator.plot()

    corpus_explorer.link_xy_range(corpus_annotator)
    corpus_explorer.link_selection("raw", corpus_annotator, "raw")
    dataset_sync_explorer(dataset, corpus_explorer, {"raw": "raw"})
    dataset_sync_explorer(dataset, corpus_annotator, {"raw": "raw"})

    sidebar = column(dataset.update_pusher, dataset.pop_table)
    layout = row(sidebar, corpus_explorer.view(), corpus_annotator.view())
    return layout


@servable(title="Snorkel Crosscheck")
def snorkel_crosscheck(dataset, lf_list, height=600, width=600):
    """
    Use the dev set to check labeling functions; use the labeling functions to hint at potential annotation.

    Layout:

    sidebar | [inspect LFs here] | [annotate here]
    """
    snorkel_explorer = BokehSnorkelExplorer(
        {"raw": dataset.dfs["raw"], "labeled": dataset.dfs["dev"]},
        title="Snorkel: square for correct, x for incorrect, + for missed, o for hit; click on legends to hide or show LF",
        height=height,
        width=width,
    )

    corpus_annotator = BokehCorpusAnnotator(
        {"raw": dataset.dfs["raw"]},
        title="Annotator: apply labels to the selected points",
        height=height,
        width=width,
    )

    snorkel_explorer.plot()
    for _lf in lf_list:
        snorkel_explorer.plot_lf(_lf)
    snorkel_explorer.figure.legend.click_policy = "hide"
    corpus_annotator.plot()

    snorkel_explorer.link_xy_range(corpus_annotator)
    snorkel_explorer.link_selection("raw", corpus_annotator, "raw")
    dataset_sync_explorer(dataset, snorkel_explorer, {"raw": "raw", "dev": "labeled"})
    dataset_sync_explorer(dataset, corpus_annotator, {"raw": "raw"})

    sidebar = column(dataset.update_pusher, dataset.pop_table)
    layout = row(sidebar, snorkel_explorer.view(), corpus_annotator.view())
    return layout


@servable(title="Active Learning")
def active_learning(dataset, vectorizer, vecnet_callback, height=600, width=600):
    """
    [MISSING SUMMARY DOCSTRING]

    Layout:

    sidebar | [inspect soft labels here] | [annotate here] | [search here]
    """
    softlabel_explorer = BokehSnorkelExplorer(
        {"raw": dataset.dfs["raw"], "labeled": dataset.dfs["dev"]},
        "pred_label",
        "pred_score",
        title="Prediction Visualizer: retrain model and locate confusions",
        height=height,
        width=width,
    )

    corpus_annotator = BokehCorpusAnnotator(
        {"raw": dataset.dfs["raw"]},
        title="Annotator: apply labels to the selected points",
        height=height,
        width=width,
    )

    corpus_explorer = BokehCorpusExplorer(
        {"raw": dataset.dfs["raw"]},
        title="Corpus: use the search widget for highlights",
        height=height,
        width=width,
    )

    softlabel_explorer.plot()
    corpus_annotator.plot()
    corpus_explorer.plot()

    softlabel_explorer.link_xy_range(corpus_annotator)
    softlabel_explorer.link_xy_range(corpus_explorer)
    softlabel_explorer.link_selection("raw", corpus_annotator, "raw")
    softlabel_explorer.link_selection("raw", corpus_explorer, "raw")
    dataset_sync_explorer(dataset, softlabel_explorer, {"raw": "raw", "dev": "labeled"})
    dataset_sync_explorer(dataset, corpus_annotator, {"raw": "raw"})
    dataset_sync_explorer(dataset, corpus_explorer, {"raw": "raw"})

    def setup_model_retrainer():
        model_retrainer = Button(label="Train model", button_type="primary")
        epochs_slider = Slider(start=1, end=20, value=1, step=1, title="# epochs")

        def retrain_model():
            """
            Callback function.
            """
            dataset.setup_label_coding()
            model = vecnet_callback()

            train_loader = dataset.loader("raw", vectorizer, smoothing_coeff=0.2)
            dev_loader = dataset.loader("dev", vectorizer)

            _ = model.train(train_loader, dev_loader, epochs=epochs_slider.value)
            logger.good("Callback 1/2: retrained model")

            for _key in ["raw", "dev"]:
                _probs = model.predict_proba(dataset.dfs[_key]["text"].tolist())
                _labels = [
                    dataset.label_decoder[_val] for _val in _probs.argmax(axis=-1)
                ]
                _scores = _probs.max(axis=-1).tolist()
                dataset.dfs[_key]["pred_label"] = pd.Series(_labels)
                dataset.dfs[_key]["pred_score"] = pd.Series(_scores)

            softlabel_explorer._update_sources()
            softlabel_explorer.plot()
            logger.good("Callback 2/2: updated predictions")

        model_retrainer.on_click(retrain_model)
        return model_retrainer, epochs_slider

    model_retrainer, epochs_slider = setup_model_retrainer()
    sidebar = column(
        model_retrainer, epochs_slider, dataset.update_pusher, dataset.pop_table
    )
    layout = row(
        sidebar,
        *[
            _plot.view()
            for _plot in [softlabel_explorer, corpus_annotator, corpus_explorer]
        ]
    )
    return layout
