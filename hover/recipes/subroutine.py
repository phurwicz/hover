"""
???+ note "Building blocks of high-level recipes."

    Includes the following:

    -   functions for creating individual standard explorers appropriate for a dataset.
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
    """
    ???+ note "Get the right `hover.core.explorer` class given a task and a feature."

        Can be useful for dynamically creating explorers without knowing the feature in advance.

        | Param     | Type  | Description                          |
        | :-------- | :---- | :----------------------------------- |
        | `task`    | `str` | name of the task, which can be `"finder"`, `"annotator"`, `"margin"`, `"softlabel"`, or `"snorkel"` |
        | `feature` | `str` | name of the main feature, which can be `"text"`, `"audio"` or `"image"` |

        Usage:
        ```python
        # this creates an instance of BokehTextFinder
        explorer = get_explorer_class("finder", "text")(*args, **kwargs)
        ```
    """
    assert task in EXPLORER_CATALOG, f"Invalid task: {task}"
    assert feature in EXPLORER_CATALOG[task], f"Invalid feature: {feature}"
    return EXPLORER_CATALOG[task][feature]


def standard_annotator(dataset, **kwargs):
    """
    ???+ note "Set up a `BokehDataAnnotator` for a `SupervisableDataset`."

        The annotator has a few standard interactions with the dataset:

        -   read all subsets of the dataset
        -   subscribe to all updates in the dataset
        -   can commit annotations through selections in the "raw" subset

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` | | kwargs to forward to the `BokehDataAnnotator` |
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

    # annotators by default link the selection for preview
    dataset.subscribe_selection_view(annotator, ["raw", "train", "dev", "test"])
    return annotator


def standard_finder(dataset, **kwargs):
    """
    ???+ note "Set up a `BokehDataFinder` for a `SupervisableDataset`."

        The finder has a few standard interactions with the dataset:

        -   read all subsets of the dataset
        -   subscribe to all updates in the dataset

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` | | kwargs to forward to the `BokehDataFinder` |
    """
    # auto-detect the (main) feature to use
    feature = dataset.__class__.FEATURE_KEY
    explorer_cls = get_explorer_class("finder", feature)

    # first "static" version of the plot
    subsets = explorer_cls.SUBSET_GLYPH_KWARGS.keys()
    finder = explorer_cls.from_dataset(
        dataset,
        {_k: _k for _k in subsets},
        title="Finder: use the search widget for highlights",
        **kwargs,
    )
    finder.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(finder, {_k: _k for _k in subsets})
    return finder


def standard_snorkel(dataset, **kwargs):
    """
    ???+ note "Set up a `BokehSnorkelExplorer` for a `SupervisableDataset`."

        The snorkel explorer has a few standard interactions with the dataset:

        -   read "raw" and "dev" subsets of the dataset, interpreting "dev" as "labeled"
        -   subscribe to all updates in those subsets

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` | | kwargs to forward to the `BokehSnorkelExplorer` |
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
    ???+ note "Set up a `BokehSoftLabelExplorer` for a `SupervisableDataset`."

        The soft label explorer has a few standard interactions with the dataset:

        -   read all subsets of the dataset
        -   subscribe to all updates in the dataset

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` | | kwargs to forward to the `BokehSoftLabelExplorer` |
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
        title="SoftLabel: inspect predictions and scores",
        **kwargs,
    )
    softlabel.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(softlabel, {_k: _k for _k in subsets})
    return softlabel
