"""
Functions commonly used by classes in this submodule.

Note that functions which are also used outside this submodule should be moved up.
"""
import hover.core.explorer as hovex


EXPLORER_CATALOG = {
    "finder": {
        "text": hovex.BokehTextFinder,
        "audio": hovex.BokehAudioFinder,
        "image": hovex.BokehImageFinder,
    },
    "annotator": {
        "text": hovex.BokehTextAnnotator,
        "audio": hovex.BokehAudioAnnotator,
        "image": hovex.BokehImageAnnotator,
    },
    "margin": {
        "text": hovex.BokehTextMargin,
        "audio": hovex.BokehAudioMargin,
        "image": hovex.BokehImageMargin,
    },
    "softlabel": {
        "text": hovex.BokehTextSoftLabel,
        "audio": hovex.BokehAudioSoftLabel,
        "image": hovex.BokehImageSoftLabel,
    },
    "snorkel": {
        "text": hovex.BokehTextSnorkel,
        "audio": hovex.BokehAudioSnorkel,
        "image": hovex.BokehImageSnorkel,
    },
}


def get_explorer_class(task, feature):
    assert task in EXPLORER_CATALOG, f"Invalid task: {task}"
    assert feature in EXPLORER_CATALOG[task], f"Invalid feature: {feature}"
    return EXPLORER_CATALOG[task][feature]


def standard_annotator(dataset, **kwargs):
    """
    Standard TextAnnotator and its interaction with a dataset.
    """
    # auto-detect the (main) feature to use
    feature = dataset.__class__.FEATURE_KEY
    explorer_cls = get_explorer_class("annotator", feature)

    # first "static" version of the plot
    subsets = explorer_cls.SUBSET_GLYPH_KWARGS.keys()
    annotator = explorer_cls.from_dataset(
        dataset,
        {_k: _k for _k in subsets},
        title="Annotator: apply labels to the selected points",
        **kwargs,
    )
    annotator.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(annotator, {_k: _k for _k in subsets})

    # annotators can commit to a dataset
    dataset.subscribe_data_commit(annotator, {"raw": "raw"})
    return annotator


def standard_finder(dataset, **kwargs):
    """
    Standard TextFinder and its interaction with a dataset.
    """
    # auto-detect the (main) feature to use
    feature = dataset.__class__.FEATURE_KEY
    explorer_cls = get_explorer_class("finder", feature)

    # first "static" version of the plot
    subsets = explorer_cls.SUBSET_GLYPH_KWARGS.keys()
    finder = explorer_cls.from_dataset(
        dataset,
        {_k: _k for _k in subsets},
        title="Text: use the search widget for highlights",
        **kwargs,
    )
    finder.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(finder, {_k: _k for _k in subsets})
    return finder


def standard_snorkel(dataset, **kwargs):
    """
    Standard SnorkelExplorer and its interaction with a dataset.
    """
    # auto-detect the (main) feature to use
    feature = dataset.__class__.FEATURE_KEY
    explorer_cls = get_explorer_class("snorkel", feature)

    # first "static" version of the plot
    snorkel = explorer_cls.from_dataset(
        dataset,
        {"raw": "raw", "dev": "labeled"},
        title="Snorkel: square for correct, x for incorrect, + for missed, o for hit; click on legends to hide or show LF",
        **kwargs,
    )
    snorkel.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(snorkel, {"raw": "raw", "dev": "labeled"})
    return snorkel


def standard_softlabel(dataset, **kwargs):
    """
    Standard SoftLabelExplorer and its interaction with a dataset.
    """
    # auto-detect the (main) feature to use
    feature = dataset.__class__.FEATURE_KEY
    explorer_cls = get_explorer_class("softlabel", feature)

    # first "static" version of the plot
    subsets = explorer_cls.SUBSET_GLYPH_KWARGS.keys()
    softlabel = explorer_cls.from_dataset(
        dataset,
        {_k: _k for _k in subsets},
        "pred_label",
        "pred_score",
        title="Prediction Visualizer: retrain model and locate confusions",
        **kwargs,
    )
    softlabel.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(softlabel, {_k: _k for _k in subsets})
    return softlabel
