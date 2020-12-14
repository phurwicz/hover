"""
Functions commonly used by classes in this submodule.

Note that functions which are also used outside this submodule should be moved up.
"""


def standard_annotator(dataset, **kwargs):
    """
    Standard annotator and its interaction with a dataset.
    """
    corpus_annotator = BokehCorpusAnnotator.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Annotator: apply labels to the selected points",
        **kwargs
    )
    corpus_annotator.plot()
    dataset.subscribe_update_push(
        corpus_annotator, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    dataset.subscribe_data_commit(corpus_annotator, {"raw": "raw"})
    return corpus_annotator


def standard_explorer(dataset, **kwargs):
    corpus_explorer = BokehCorpusExplorer.from_dataset(
        dataset,
        {_k: _k for _k in ["raw", "train", "dev", "test"]},
        title="Corpus: use the search widget for highlights",
        height=height,
        width=width,
    )
    corpus_explorer.plot()
    dataset.subscribe_update_push(
        corpus_explorer, {_k: _k for _k in ["raw", "train", "dev", "test"]}
    )
    return corpus_explorer
