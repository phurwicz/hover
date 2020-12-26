"""
Functions commonly used by classes in this submodule.

Note that functions which are also used outside this submodule should be moved up.
"""
from hover.core.explorer import (
    BokehCorpusFinder,
    BokehCorpusAnnotator,
    BokehCorpusSoftLabel,
    BokehCorpusSnorkel,
)


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


def standard_finder(dataset, **kwargs):
    """
    Standard CorpusFinder and its interaction with a dataset.
    """
    # first "static" version of the plot
    corpus_finder = BokehCorpusFinder.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Corpus: use the search widget for highlights",
        **kwargs
    )
    corpus_finder.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(
        corpus_finder, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    return corpus_finder


def standard_snorkel(dataset, **kwargs):
    """
    Standard SnorkelExplorer and its interaction with a dataset.
    """
    # first "static" version of the plot
    corpus_snorkel = BokehCorpusSnorkel.from_dataset(
        dataset,
        {"raw": "raw", "dev": "labeled"},
        title="Snorkel: square for correct, x for incorrect, + for missed, o for hit; click on legends to hide or show LF",
        **kwargs
    )
    corpus_snorkel.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(corpus_snorkel, {"raw": "raw", "dev": "labeled"})
    return corpus_snorkel


def standard_softlabel(dataset, **kwargs):
    """
    Standard SoftLabelExplorer and its interaction with a dataset.
    """
    # first "static" version of the plot
    corpus_softlabel = BokehCorpusSoftLabel.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev"]},
        "pred_label",
        "pred_score",
        title="Prediction Visualizer: retrain model and locate confusions",
        **kwargs
    )
    corpus_softlabel.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(
        corpus_softlabel, {_k: _k for _k in ["raw", "train", "dev"]}
    )
    return corpus_softlabel
