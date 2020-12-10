"""
Dataset objects which extend beyond DataFrames.
Specifically, we need a collection of DataFrames where rows can be transferred cleanly and columns can be transformed easily.
"""
from abc import ABC
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich.console import Console
from hover import module_config
from bokeh.models import Button, Dropdown, ColumnDataSource, DataTable, TableColumn
from bokeh.layouts import row


console = Console()


class SupervisableDataset(ABC):
    """
    Feature-agnostic class for a dataset open to supervision.
    Raw -- piecewise annoatation -> Gold -> Dev/Test
    Raw -- batch annotation -> Noisy -> Train

    Keeping a DataFrame form and a list-of-dicts ("dictl") form, with the intention that
    - the DataFrame form supports most kinds of operations;
    - the list-of-dicts form could be useful for manipulations outside the scope of pandas;
    - synchronization between the two forms should be called sparingly.
    """

    ORDERED_SUBSET = ("test", "dev", "train", "raw")
    FEATURE_KEY = "feature"

    def __init__(
        self,
        raw_dictl,
        train_dictl=None,
        dev_dictl=None,
        test_dictl=None,
        feature_key="feature",
        label_key="label",
    ):
        """
        Initialize the dataset with dictl and df forms; initialize the mapping between categorical-int and string labels.

        - param raw_dictl: a list of dicts holding the raw data that DO NOT have annotation.
        - param train_dictl: a list of dicts holding the batch-annotated noisy train set.
        - param dev_dictl: a list of dicts holding the gold dev set.
        - param test_dictl: a list of dicts holding the gold test set.
        - param feature_key: key in each piece of dict mapping to the feature.
        - param label_key: key in each piece of dict mapping to the ground truth in STRING form.
        """

        def dictl_transform(dictl, labels=True):
            """
            Burner function to transform the input list of dictionaries into standard format.
            """
            # edge case when dictl is empty or None
            if not dictl:
                return []

            # transform the feature and possibly the label
            key_transform = {feature_key: self.__class__.FEATURE_KEY}
            if labels:
                key_transform[label_key] = "label"

            def burner(d):
                """
                Burner function to transform a single dict.
                """
                if labels:
                    assert label_key in d, f"Expected dict key {label_key}"

                trans_d = {key_transform.get(_k, _k): _v for _k, _v in d.items()}

                if not labels:
                    trans_d["label"] = module_config.ABSTAIN_DECODED

                return trans_d

            return [burner(_d) for _d in dictl]

        self.dictls = {
            "raw": dictl_transform(raw_dictl, labels=False),
            "train": dictl_transform(train_dictl),
            "dev": dictl_transform(dev_dictl),
            "test": dictl_transform(test_dictl),
        }

        self.synchronize_dictl_to_df()
        self.df_deduplicate()
        self.synchronize_df_to_dictl()
        self.setup_widgets()
        # self.setup_label_coding() # redundant if setup_pop_table() immediately calls this again
        self.setup_pop_table()

    def setup_widgets(self):
        """
        Critical widgets for interactive data management.
        """
        self.update_pusher = Button(
            label="Push", button_type="success", height_policy="fit", width_policy="min"
        )
        self.data_committer = Dropdown(
            label="Commit",
            button_type="warning",
            menu=["train", "dev", "test"],
            height_policy="fit",
            width_policy="min",
        )
        self.dedup_trigger = Button(
            label="Dedup",
            button_type="warning",
            height_policy="fit",
            width_policy="min",
        )

        def callback_dedup():
            self.deduplicate()

        self.dedup_trigger.on_click(callback_dedup)

    def layout_widgets(self):
        return column(
            row(self.update_pusher, self.data_committer, self.dedup_trigger),
            self.pop_table,
        )

    def subscribe_update_push(self, explorer, subset_mapping):
        """
        Enable pushing updated DataFrames to explorers that depend on them.

        Note: the reason we need this is due to `self.dfs[key] = ...`-like assignments. If DF operations were all in-place, then the explorers could directly access the updates through their `self.dfs` references.
        """
        assert isinstance(explorer, BokehForLabeledText)

        def callback_push():
            df_dict = {_v: self.dfs[_k] for _k, _v in subset_mapping.items()}
            explorer._setup_dfs(df_dict)
            explorer._update_sources()

        self.update_pusher.on_click(callback_push)
        console.print(
            f"Subscribed {explorer.__class__.__name__} to dataset pushes: {subset_mapping}",
            style="green",
        )

    def subscribe_data_commit(self, explorer, subset_mapping):
        """
        Enable committing data across subsets, specified by a selection in an explorer and a dropdown widget of the dataset.
        """
        if self.data_committer.subscribed_events:
            console.print(
                f"Attempting subscribe_data_commit: found existing callback, and there should only be one such callback",
                style="red",
            )
            return

        def callback_commit(event):
            for sub_k, sub_v in subset_mapping.items():
                sub_to = event.item
                select_idx = explorer.sources[sub_v].selected.indices
                if not selected_idx:
                    console.print(
                        "Attempting data commit: did not select any data points.",
                        style="yellow",
                    )
                    return
                selected_slice = self.dfs[sub_k].at[selected_idx]
                self.dfs[sub_to] = pd.concat(
                    [self.dfs[sub_to], selected_slice], axis=0, sort=False
                )
                self.dfs[sub_to].drop_duplicates(
                    subset=[self.__class__.FEATURE_KEY], keep="first", inplace=True
                )
                console.print(
                    f"Committed {selected_slice.shape[0]} entries from [{sub_k}] to [{sub_to}].",
                    style="blue",
                )

        self.data_committer.on_click(callback_commit)
        console.print(
            f"Subscribed {explorer.__class__.__name__} to dataset commits: {subset_mapping}",
            style="green",
        )

    def setup_label_coding(self):
        """
        Auto-determine labels in the dataset, then create encoder/decoder in lexical order.
        Add ABSTAIN as a no-label placeholder.
        """
        all_labels = set()
        for _key in self.__class__.ORDERED_SUBSET[:-1]:
            _df = self.dfs[_key]
            if _df.empty:
                continue
            assert "label" in _df.columns
            _found_labels = set(_df["label"].tolist())
            all_labels = all_labels.union(_found_labels)

        # exclude ABSTAIN from self.classes, but include it in the encoding
        all_labels.discard(module_config.ABSTAIN_DECODED)
        self.classes = sorted(all_labels)
        self.label_encoder = {
            **{_label: _i for _i, _label in enumerate(self.classes)},
            module_config.ABSTAIN_DECODED: module_config.ABSTAIN_ENCODED,
        }
        self.label_decoder = {_v: _k for _k, _v in self.label_encoder.items()}

        console.print(
            f"Set up label encoder/decoder with {len(self.classes)} classes.",
            style="green",
        )
        self.validate_labels()

    def validate_labels(self, raise_exception=True):
        """
        Check that every label is in the encoder.
        """
        for _key in self.__class__.ORDERED_SUBSET[:-1]:
            _df = self.dfs[_key]
            _invalid_indices = None
            if _df.empty:
                continue
            assert "label" in _df.columns
            _mask = _df["label"].apply(lambda x: x in self.label_encoder)
            _invalid_indices = np.where(_mask == False)[0].tolist()
            if _invalid_indices:
                console.print(f"Subset [{_key}] has invalid labels:")
                console.print({_df.loc[_invalid_indices]})
                if raise_exception:
                    raise ValueError("invalid labels")

    def setup_pop_table(self, subsets=("test", "dev", "train", "raw"), **kwargs):
        pop_source = ColumnDataSource(dict())
        pop_columns = [
            TableColumn(field="label", title="label"),
            *[
                TableColumn(field=f"count_{_subset}", title=_subset)
                for _subset in subsets
            ],
        ]
        self.pop_table = DataTable(source=pop_source, columns=pop_columns, **kwargs)

        def update_population():
            """
            Callback function.
            """
            # make sure that the label coding is correct
            self.setup_label_coding()

            # re-compute label population
            pop_data = dict(label=self.classes)
            for _subset in subsets:
                _df = self.dfs[_subset]
                _subpop = _df["label"].value_counts()
                pop_data[f"count_{_subset}"] = [
                    _subpop.get(_label, 0) for _label in self.classes
                ]

            # push results to bokeh data source
            pop_source.data = pop_data

            # send a console message
            console.print(
                f"Pop updater: latest population with {len(self.classes)} classes",
                style="green",
            )

        update_population()
        self.update_pusher.on_click(update_population)

    def df_deduplicate(self):
        """
        Cross-deduplicate data entries by feature between subsets.
        """
        # for data entry accounting
        before, after = dict(), dict()

        # keep track of which df has which columns and which rows came from which subset
        columns = dict()
        for _key in self.__class__.ORDERED_SUBSET:
            before[_key] = self.dfs[_key].shape[0]
            columns[_key] = self.dfs[_key].columns
            self.dfs[_key]["__subset"] = _key

        # concatenate in order and deduplicate
        overall_df = pd.concat(
            [self.dfs[_key] for _key in self.__class__.ORDERED_SUBSET],
            axis=0,
            sort=False,
        )
        overall_df.drop_duplicates(
            subset=[self.__class__.FEATURE_KEY], keep="first", inplace=True
        )
        overall_df.reset_index(drop=True, inplace=True)

        # cut up slices
        for _key in self.__class__.ORDERED_SUBSET:
            self.dfs[_key] = overall_df[overall_df["__subset"] == _key].reset_index(
                drop=True, inplace=False
            )[columns[_key]]
            after[_key] = self.dfs[_key].shape[0]
            console.print(
                f"--subset {_key} rows: {before[_key]} -> {after[_key]}.", style="blue"
            )

    def synchronize_dictl_to_df(self):
        """
        Re-make dataframes from lists of dictionaries.
        """
        self.dfs = dict()
        for _key, _dictl in self.dictls.items():
            if _dictl:
                _df = pd.DataFrame(_dictl)
                assert self.__class__.FEATURE_KEY in _df.columns
                assert "label" in _df.columns
            else:
                _df = pd.DataFrame(columns=[self.__class__.FEATURE_KEY, "label"])

            self.dfs[_key] = _df

    def synchronize_df_to_dictl(self):
        """
        Re-make lists of dictionaries from dataframes.
        """
        self.dictls = dict()
        for _key, _df in self.dfs.items():
            self.dictls[_key] = _df.to_dict(orient="records")

    def compute_2d_embedding(self, vectorizer, method, **kwargs):
        """
        Get embeddings in the xy-plane and return the reducer.
        """
        from hover.core.representation.reduction import DimensionalityReducer

        # prepare input vectors to manifold learning
        subset = ["raw", "train", "dev"]
        fit_inp = []
        for _key in subset:
            _df = self.dfs[_key]
            if _df.empty:
                continue
            fit_inp += _df[self.__class__.FEATURE_KEY].tolist()
        fit_arr = np.array([vectorizer(_inp) for _inp in tqdm(fit_inp)])

        # initialize and fit manifold learning reducer
        reducer = DimensionalityReducer(fit_arr)
        embedding = reducer.fit_transform(method, **kwargs)

        # assign x and y coordinates to dataset
        start_idx = 0
        for _key in subset:
            _df = self.dfs[_key]
            _length = _df.shape[0]
            _df["x"] = pd.Series(embedding[start_idx : (start_idx + _length), 0])
            _df["y"] = pd.Series(embedding[start_idx : (start_idx + _length), 1])
            start_idx += _length

        return reducer

    def loader(self, key, vectorizer, batch_size=64, smoothing_coeff=0.0):
        """
        Prepare a Torch Dataloader for training or evaluation.

        - param key(str): the subset of dataset to use.
        - param vectorizer(callable): callable that turns a string into a vector.
        - param smoothing_coeff: the smoothing coeffient for soft labels.
          - type smoothing_coeff: float
        """
        # lazy import: missing torch should not break the rest of the class
        from hover.utils.torch_helper import vector_dataloader, one_hot, label_smoothing

        # take the slice that has a meaningful label
        df = self.dfs[key][self.dfs[key]["label"] != module_config.ABSTAIN_DECODED]

        labels = df["label"].apply(lambda x: self.label_encoder[x]).tolist()
        features = df[self.__class__.FEATURE_KEY].tolist()
        output_vectors = one_hot(labels, num_classes=len(self.classes))

        console.print(f"Preparing {key} input vectors...", style="blue")
        input_vectors = [vectorizer(_f) for _f in tqdm(features)]
        if smoothing_coeff > 0.0:
            output_vectors = label_smoothing(
                output_vectors, coefficient=smoothing_coeff
            )
        console.print(f"Preparing {key} data loader...", style="blue")
        loader = vector_dataloader(input_vectors, output_vectors, batch_size=batch_size)
        console.print(
            f"Prepared {key} loader consisting of {len(features)} examples with batch size {batch_size}",
            style="green",
        )
        return loader


class SupervisableTextDataset(SupervisableDataset):
    """
    Can add text-specific methods.
    """

    FEATURE_KEY = "text"
