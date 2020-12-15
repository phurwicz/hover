"""
Functions commonly used by classes in this submodule.

Note that functions which are also used outside this submodule should be moved up.
"""


def standard_annotator(dataset, **kwargs):
    """
    Standard CorpusAnnotator and its interaction with a dataset.
    """
    # first "static" version of the plot
    corpus_annotator = BokehCorpusAnnotator.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Annotator: apply labels to the selected points",
        **kwargs
    )
    corpus_annotator.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(
        corpus_annotator, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )

    # annotators can commit to a dataset
    dataset.subscribe_data_commit(corpus_annotator, {"raw": "raw"})
    return corpus_annotator


def standard_explorer(dataset, **kwargs):
    """
    Standard CorpusExplorer and its interaction with a dataset.
    """
    # first "static" version of the plot
    corpus_explorer = BokehCorpusExplorer.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Corpus: use the search widget for highlights",
        height=height,
        width=width,
    )
    corpus_explorer.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(
        corpus_explorer, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    return corpus_explorer


def standard_snorkel(dataset, **kwargs):
    """
    Standard SnorkelExplorer and its interaction with a dataset.
    """
    # first "static" version of the plot
    snorkel_explorer = BokehSnorkelExplorer.from_dataset(
        dataset,
        {"raw": "raw", "dev": "labeled"},
        title="Snorkel: square for correct, x for incorrect, + for missed, o for hit; click on legends to hide or show LF",
        height=height,
        width=width,
    )
    snorkel_explorer.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(snorkel_explorer, {"raw": "raw", "dev": "labeled"})
    return snorkel_explorer


def standard_softlabel(dataset, **kwargs):
    """
    Standard SoftLabelExplorer and its interaction with a dataset.
    """
    # first "static" version of the plot
    softlabel_explorer = BokehSoftLabelExplorer.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev"]},
        "pred_label",
        "pred_score",
        title="Prediction Visualizer: retrain model and locate confusions",
        height=height,
        width=width,
    )
    softlabel_explorer.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(
        softlabel_explorer, {_k: _k for _k in ["raw", "train", "dev"]}
    )
    return softlabel_explorer
