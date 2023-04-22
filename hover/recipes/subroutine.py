"""
???+ note "Building blocks of high-level recipes."

    Includes the following:

    -   functions for creating individual standard explorers appropriate for a dataset.
"""
import re
import numpy as np
import hover.core.explorer as hovex
from bokeh.layouts import row, column
from bokeh.models import Button
from rich.console import Console
from .local_config import DEFAULT_REDUCTION_METHOD


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


def recipe_layout(*components, style="horizontal"):
    """
    ???+ note "Create a recipe-level layout of bokeh objects."

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `*components` | `bokeh` objects | objects to be plotted      |
        | `style`    | `str`    | "horizontal" or "vertical"           |
    """
    if style == "horizontal":
        return row(*components)
    elif style == "vertical":
        return column(*components)
    else:
        raise ValueError(f"Unexpected layout style {style}")


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
        title="Annotator: apply labels to selected RAW points",
        **kwargs,
    )
    annotator.activate_search()
    annotator.plot()

    # subscribe for dataset updates
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
        title="Finder: use search for highlight and filter",
        **kwargs,
    )
    finder.activate_search()
    finder.plot()

    # subscribe for dataset updates
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
        title="Snorkel: □ for correct, x for incorrect, + for missed, o for hit; click on legends to hide or show LF",
        **kwargs,
    )
    snorkel.activate_search()
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
        | `**kwargs` | | kwargs to forward to `BokehSoftLabelExplorer` |
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
        title="SoftLabel: inspect predictions and use score range as filter",
        **kwargs,
    )
    softlabel.activate_search()
    softlabel.plot()

    # subscribe to dataset widgets
    dataset.subscribe_update_push(softlabel, {_k: _k for _k in subsets})
    return softlabel


def active_learning_components(dataset, vecnet, **kwargs):
    """
    ???+ note "Active-learning specific components of a recipe."

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `vecnet`   | `VectorNet` | vecnet to use in the loop          |
        | `**kwargs` | | kwargs to forward to the `BokehSoftLabelExplorer` |
    """
    console = Console()
    softlabel = standard_softlabel(dataset, **kwargs)
    feature_key = dataset.__class__.FEATURE_KEY

    # patch coordinates for representational similarity analysis
    # some datasets may have multiple embeddings; use the one with lowest dimension
    embedding_cols = sorted(softlabel.find_embedding_fields())
    manifold_dim, _ = re.findall(r"\d+", embedding_cols[0])
    manifold_dim = int(manifold_dim)
    manifold_traj_cols = embedding_cols[:manifold_dim]
    for _col in manifold_traj_cols:
        _total_dim, _ = re.findall(r"\d+", _col)
        _total_dim = int(_total_dim)
        assert (
            _total_dim == manifold_dim
        ), f"Dim mismatch: {_total_dim} vs. {manifold_dim}"
        softlabel.value_patch_by_slider(
            _col, f"{_col}_traj", title="Manifold trajectory step"
        )

    # recipe-specific widget
    model_trainer = Button(label="Train model", button_type="primary")

    def retrain_vecnet():
        """
        Callback subfunction 1 of 2.
        """
        model_trainer.disabled = True
        console.print("Start training... button will be disabled temporarily.")
        dataset.setup_label_coding()
        vecnet.auto_adjust_setup(dataset.classes)

        train_loader = vecnet.prepare_loader(dataset, "train", smoothing_coeff=0.2)
        if dataset.subset("dev").shape[0] > 0:
            dev_loader = vecnet.prepare_loader(dataset, "dev")
        else:
            dataset._warn("dev set is empty, borrowing train set for validation.")
            dev_loader = train_loader

        _ = vecnet.train(train_loader, dev_loader)
        vecnet.save()
        console.print("-- 1/2: retrained vecnet")

    def update_softlabel_plot():
        """
        Callback subfunction 2 of 2.
        """
        # combine inputs and compute outputs of all non-test subsets
        use_subsets = ("raw", "train", "dev")
        inps = []
        for _key in use_subsets:
            inps.extend(dataset.subset(_key)[feature_key].tolist())

        probs = vecnet.predict_proba(inps)
        labels = [dataset.label_decoder[_val] for _val in probs.argmax(axis=-1)]
        scores = probs.max(axis=-1).tolist()
        traj_arr, _, _ = vecnet.manifold_trajectory(
            inps,
            method=DEFAULT_REDUCTION_METHOD,
            reducer_kwargs=dict(dimension=manifold_dim),
            spline_kwargs=dict(points_per_step=5),
        )

        offset = 0
        for _key in use_subsets:
            _length = dataset.subset(_key).shape[0]
            # skip subset if empty
            if _length == 0:
                continue
            _slice = slice(offset, offset + _length)
            dataset.subset(_key)["pred_label"] = labels[_slice]
            dataset.subset(_key)["pred_score"] = scores[_slice]
            for i, _col in enumerate(manifold_traj_cols):
                # all steps, selected slice
                _traj = traj_arr[:, _slice, i]
                # selected slice, all steps
                _traj = list(np.swapaxes(_traj, 0, 1))
                dataset.subset(_key)[f"{_col}_traj"] = _traj

            offset += _length

        softlabel._dynamic_callbacks["adjust_patch_slider"]()
        softlabel._update_sources()
        model_trainer.disabled = False
        console.print("-- 2/2: updated predictions. Training button is re-enabled.")

    def callback_sequence():
        """
        Overall callback function.
        """
        retrain_vecnet()
        update_softlabel_plot()

    model_trainer.on_click(callback_sequence)

    return softlabel, model_trainer
