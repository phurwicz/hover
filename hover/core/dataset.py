"""
???+ note "Dataset classes which extend beyond DataFrames."

    When we supervise a collection of data, these operations need to be simple:

    -   managing `raw`/`train`/`dev`/`test` subsets
    -   transferring data points between subsets
    -   pulling updates from annotation interfaces
    -   pushing updates to annotation interfaces
    -   getting a 2D embedding
    -   loading data for training models
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from hover import module_config
from hover.core import Loggable
from hover.utils.bokeh_helper import auto_label_color
from hover.utils.misc import current_time
from bokeh.models import (
    Button,
    Dropdown,
    ColumnDataSource,
    DataTable,
    TableColumn,
    HTMLTemplateFormatter,
)
from .local_config import (
    dataset_help_widget,
    COLOR_GLYPH_TEMPLATE,
    DATASET_SUBSET_FIELD,
)


class SupervisableDataset(Loggable):
    """
    ???+ note "Feature-agnostic class for a dataset open to supervision."

        Keeping a DataFrame form and a list-of-dicts ("dictl") form, with the intention that

        - the DataFrame form supports most kinds of operations;
        - the list-of-dicts form could be useful for manipulations outside the scope of pandas;
        - synchronization between the two forms should be called sparingly.
    """

    # 'scratch': intended to be directly editable by other objects, i.e. Explorers
    # labels will be stored but not used for information in hover itself
    SCRATCH_SUBSETS = tuple(["raw"])

    # non-'scratch': intended to be read-only outside of the class
    # 'public': labels will be considered as part of the classification task and will be used for built-in supervision
    PUBLIC_SUBSETS = tuple(["train", "dev"])
    # 'private': labels will be considered as part of the classification task and will NOT be used for supervision
    PRIVATE_SUBSETS = tuple(["test"])

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
        ???+ note "Create (1) dictl and df forms and (2) the mapping between categorical and string labels."
            | Param         | Type   | Description                          |
            | :------------ | :----- | :----------------------------------- |
            | `raw_dictl`   | `list` | list of dicts holding the **to-be-supervised** raw data |
            | `train_dictl` | `list` | list of dicts holding any **supervised** train data |
            | `dev_dictl`   | `list` | list of dicts holding any **supervised** dev data   |
            | `test_dictl`  | `list` | list of dicts holding any **supervised** test data  |
            | `feature_key` | `str`  | the key for the feature in each piece of data |
            | `label_key`   | `str`  | the key for the `**str**` label in supervised data |
        """
        self._info("Initializing...")

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
        self.setup_file_export()
        self.setup_pop_table(width_policy="fit", height_policy="fit")
        self.setup_sel_table(width_policy="fit", height_policy="fit")
        self._good(f"{self.__class__.__name__}: finished initialization.")

    def copy(self, use_df=True):
        """
        ???+ note "Create another instance, copying over the data entries."
            | Param    | Type   | Description                          |
            | :------- | :----- | :----------------------------------- |
            | `use_df` | `bool` | whether to use the df or dictl form  |
        """
        if use_df:
            self.synchronize_df_to_dictl()
        return self.__class__(
            raw_dictl=self.dictls["raw"],
            train_dictl=self.dictls["train"],
            dev_dictl=self.dictls["dev"],
            test_dictl=self.dictls["test"],
            feature_key=self.__class__.FEATURE_KEY,
            label_key="label",
        )

    def to_pandas(self, use_df=True):
        """
        ???+ note "Export to a pandas DataFrame."
            | Param    | Type   | Description                          |
            | :------- | :----- | :----------------------------------- |
            | `use_df` | `bool` | whether to use the df or dictl form  |
        """
        if not use_df:
            self.synchronize_dictl_to_df()
        dfs = []
        for _subset in ["raw", "train", "dev", "test"]:
            _df = self.dfs[_subset].copy()
            _df[DATASET_SUBSET_FIELD] = _subset
            dfs.append(_df)

        return pd.concat(dfs, axis=0)

    @classmethod
    def from_pandas(cls, df, **kwargs):
        """
        ???+ note "Import from a pandas DataFrame."
            | Param    | Type   | Description                          |
            | :------- | :----- | :----------------------------------- |
            | `df` | `DataFrame` | with a "SUBSET" field dividing subsets |
        """
        dictls = {}
        for _subset in ["raw", "train", "dev", "test"]:
            dictls[_subset] = df[df[DATASET_SUBSET_FIELD] == _subset].to_dict(
                orient="records"
            )
        return cls(
            raw_dictl=dictls["raw"],
            train_dictl=dictls["train"],
            dev_dictl=dictls["dev"],
            test_dictl=dictls["test"],
            **kwargs,
        )

    def setup_widgets(self):
        """
        ???+ note "Create `bokeh` widgets for interactive data management."
        """
        self.update_pusher = Button(
            label="Push", button_type="success", height_policy="fit", width_policy="min"
        )
        self.data_committer = Dropdown(
            label="Commit",
            button_type="warning",
            menu=[*self.__class__.PUBLIC_SUBSETS, *self.__class__.PRIVATE_SUBSETS],
            height_policy="fit",
            width_policy="min",
        )
        self.dedup_trigger = Button(
            label="Dedup",
            button_type="warning",
            height_policy="fit",
            width_policy="min",
        )
        self.selection_viewer = Button(
            label="View Selected",
            button_type="primary",
            height_policy="fit",
            width_policy="min",
        )

        def commit_base_callback():
            """
            COMMIT creates cross-duplicates between subsets.

            - PUSH shall be blocked until DEDUP is executed.
            """
            self.dedup_trigger.disabled = False
            self.update_pusher.disabled = True

        def dedup_base_callback():
            """
            DEDUP re-creates dfs with different indices than before.

            - COMMIT shall be blocked until PUSH is executed.
            """
            self.update_pusher.disabled = False
            self.data_committer.disabled = True
            self.df_deduplicate()

        def push_base_callback():
            """
            PUSH enforces df consistency with all linked explorers.

            - DEDUP could be blocked because it stays trivial until COMMIT is executed.
            """
            self.data_committer.disabled = False
            self.dedup_trigger.disabled = True

        self.update_pusher.on_click(push_base_callback)
        self.data_committer.on_click(commit_base_callback)
        self.dedup_trigger.on_click(dedup_base_callback)

        self.help_div = dataset_help_widget()

    def view(self):
        """
        ???+ note "Defines the layout of `bokeh` objects when visualized."
        """
        # local import to avoid naming confusion/conflicts
        from bokeh.layouts import row, column

        return column(
            self.help_div,
            row(
                self.update_pusher,
                self.data_committer,
                self.dedup_trigger,
                self.selection_viewer,
                self.file_exporter,
            ),
            self.pop_table,
            self.sel_table,
        )

    def subscribe_update_push(self, explorer, subset_mapping):
        """
        ???+ note "Enable pushing updated DataFrames to explorers that depend on them."
            | Param            | Type   | Description                            |
            | :--------------- | :----- | :------------------------------------- |
            | `explorer`       | `BokehBaseExplorer` | the explorer to register  |
            | `subset_mapping` | `dict` | `dataset` -> `explorer` subset mapping |

            Note: the reason we need this is due to `self.dfs[key] = ...`-like assignments. If DF operations were all in-place, then the explorers could directly access the updates through their `self.dfs` references.
        """
        # local import to avoid import cycles
        from hover.core.explorer.base import BokehBaseExplorer

        assert isinstance(explorer, BokehBaseExplorer)

        def callback_push():
            df_dict = {_v: self.dfs[_k] for _k, _v in subset_mapping.items()}
            explorer._setup_dfs(df_dict)
            explorer._update_sources()

        self.update_pusher.on_click(callback_push)
        self._good(
            f"Subscribed {explorer.__class__.__name__} to dataset pushes: {subset_mapping}"
        )

    def subscribe_data_commit(self, explorer, subset_mapping):
        """
        ???+ note "Enable committing data across subsets, specified by a selection in an explorer and a dropdown widget of the dataset."
            | Param            | Type   | Description                            |
            | :--------------- | :----- | :------------------------------------- |
            | `explorer`       | `BokehBaseExplorer` | the explorer to register  |
            | `subset_mapping` | `dict` | `dataset` -> `explorer` subset mapping |
        """

        def callback_commit(event):
            for sub_k, sub_v in subset_mapping.items():
                sub_to = event.item
                selected_idx = explorer.sources[sub_v].selected.indices
                if not selected_idx:
                    self._warn(
                        f"Attempting data commit: did not select any data points in subset {sub_v}."
                    )
                    return

                # take selected slice, ignoring ABSTAIN'ed rows
                # CAUTION: applying selected_idx from explorer.source to self.df
                #     this assumes that the source and the df have consistent entries.
                # Consider this:
                #    keep_cols = self.dfs[sub_k].columns
                #    sel_slice = explorer.dfs[sub_v].iloc[selected_idx][keep_cols]
                sel_slice = self.dfs[sub_k].iloc[selected_idx]
                valid_slice = sel_slice[
                    sel_slice["label"] != module_config.ABSTAIN_DECODED
                ]

                # concat to the end and do some accounting
                size_before = self.dfs[sub_to].shape[0]
                self.dfs[sub_to] = pd.concat(
                    [self.dfs[sub_to], valid_slice],
                    axis=0,
                    sort=False,
                    ignore_index=True,
                )
                size_mid = self.dfs[sub_to].shape[0]
                self.dfs[sub_to].drop_duplicates(
                    subset=[self.__class__.FEATURE_KEY], keep="last", inplace=True
                )
                size_after = self.dfs[sub_to].shape[0]

                self._info(
                    f"Committed {valid_slice.shape[0]} (valid out of {sel_slice.shape[0]} selected) entries from {sub_k} to {sub_to} ({size_before} -> {size_after} with {size_mid-size_after} overwrites)."
                )
            # chain another callback
            self._callback_update_population()

        self.data_committer.on_click(callback_commit)
        self._good(
            f"Subscribed {explorer.__class__.__name__} to dataset commits: {subset_mapping}"
        )

    def subscribe_selection_view(self, explorer, subsets):
        """
        ???+ note "Enable viewing groups of data entries, specified by a selection in an explorer."
            | Param            | Type   | Description                            |
            | :--------------- | :----- | :------------------------------------- |
            | `explorer`       | `BokehBaseExplorer` | the explorer to register  |
            | `subsets`        | `list` | subset selections to consider          |
        """
        assert (
            isinstance(subsets, list) and len(subsets) > 0
        ), "Expected a non-empty list of subsets"

        def callback_view():
            sel_slices = []
            for subset in subsets:
                selected_idx = explorer.sources[subset].selected.indices
                sub_slice = explorer.dfs[subset].iloc[selected_idx]
                sel_slices.append(sub_slice)

            selected = pd.concat(sel_slices, axis=0)

            # replace this with an actual display (and analysis)
            self._callback_update_selection(selected)

        self.selection_viewer.on_click(callback_view)
        self._good(
            f"Subscribed {explorer.__class__.__name__} to selection view: {subsets}"
        )

    def setup_label_coding(self, verbose=True, debug=False):
        """
        ???+ note "Auto-determine labels in the dataset, then create encoder/decoder in lexical order."
            Add `"ABSTAIN"` as a no-label placeholder which gets ignored categorically.

            | Param     | Type   | Description                        |
            | :-------- | :----- | :--------------------------------- |
            | `verbose` | `bool` | whether to log verbosely           |
            | `debug`   | `bool` | whether to enable label validation |
        """
        all_labels = set()
        for _key in [*self.__class__.PUBLIC_SUBSETS, *self.__class__.PRIVATE_SUBSETS]:
            _df = self.dfs[_key]
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

        if verbose:
            self._good(
                f"Set up label encoder/decoder with {len(self.classes)} classes."
            )
        if debug:
            self.validate_labels()

    def validate_labels(self, raise_exception=True):
        """
        ???+ note "Assert that every label is in the encoder."

            | Param             | Type   | Description                         |
            | :---------------- | :----- | :---------------------------------- |
            | `raise_exception` | `bool` | whether to raise errors when failed |
        """
        for _key in [*self.__class__.PUBLIC_SUBSETS, *self.__class__.PRIVATE_SUBSETS]:
            _invalid_indices = None
            assert "label" in self.dfs[_key].columns
            _mask = self.dfs[_key]["label"].apply(lambda x: x in self.label_encoder)
            _invalid_indices = np.where(_mask is False)[0].tolist()
            if _invalid_indices:
                self._fail(f"Subset {_key} has invalid labels:")
                self._print({self.dfs[_key].loc[_invalid_indices]})
                if raise_exception:
                    raise ValueError("invalid labels")

    def setup_file_export(self):
        self.file_exporter = Dropdown(
            label="Export",
            button_type="warning",
            menu=["Excel", "CSV", "JSON", "pickle"],
            height_policy="fit",
            width_policy="min",
        )

        def callback_export(event, path_root=None):
            """
            A callback on clicking the 'self.annotator_export' button.
            Saves the dataframe to a pickle.
            """
            export_format = event.item

            # auto-determine the export path root
            if path_root is None:
                timestamp = current_time("%Y%m%d%H%M%S")
                path_root = f"hover-dataset-export-{timestamp}"

            export_df = self.to_pandas(use_df=True)

            if export_format == "Excel":
                export_path = f"{path_root}.xlsx"
                export_df.to_excel(export_path, index=False)
            elif export_format == "CSV":
                export_path = f"{path_root}.csv"
                export_df.to_csv(export_path, index=False)
            elif export_format == "JSON":
                export_path = f"{path_root}.json"
                export_df.to_json(export_path, orient="records")
            elif export_format == "pickle":
                export_path = f"{path_root}.pkl"
                export_df.to_pickle(export_path)
            else:
                raise ValueError(f"Unexpected export format {export_format}")

            self._good(f"Saved DataFrame to {export_path}")

        # assign the callback, keeping its reference
        self._callback_export = callback_export
        self.file_exporter.on_click(self._callback_export)

    def setup_pop_table(self, **kwargs):
        """
        ???+ note "Set up a bokeh `DataTable` widget for monitoring subset data populations."

            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `**kwargs` |        | forwarded to the `DataTable` |
        """
        subsets = [
            *self.__class__.SCRATCH_SUBSETS,
            *self.__class__.PUBLIC_SUBSETS,
            *self.__class__.PRIVATE_SUBSETS,
        ]
        pop_source = ColumnDataSource(dict())
        pop_columns = [
            TableColumn(field="label", title="label"),
            *[
                TableColumn(field=f"count_{_subset}", title=_subset)
                for _subset in subsets
            ],
            TableColumn(
                field="color",
                title="color",
                formatter=HTMLTemplateFormatter(template=COLOR_GLYPH_TEMPLATE),
            ),
        ]
        self.pop_table = DataTable(source=pop_source, columns=pop_columns, **kwargs)

        def update_population():
            """
            Callback function.
            """
            # make sure that the label coding is correct
            self.setup_label_coding()

            # re-compute label population
            eff_labels = [module_config.ABSTAIN_DECODED, *self.classes]
            color_dict = auto_label_color(self.classes)
            eff_colors = [color_dict[_label] for _label in eff_labels]

            pop_data = dict(color=eff_colors, label=eff_labels)
            for _subset in subsets:
                _subpop = self.dfs[_subset]["label"].value_counts()
                pop_data[f"count_{_subset}"] = [
                    _subpop.get(_label, 0) for _label in eff_labels
                ]

            # push results to bokeh data source
            pop_source.data = pop_data

            self._good(
                f"Population updater: latest population with {len(self.classes)} classes."
            )

        update_population()
        self.dedup_trigger.on_click(update_population)

        # store the callback so that it can be referenced by other methods
        self._callback_update_population = update_population

    def setup_sel_table(self, **kwargs):
        """
        ???+ note "Set up a bokeh `DataTable` widget for viewing selected data points."

            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `**kwargs` |        | forwarded to the `DataTable` |
        """
        sel_source = ColumnDataSource(dict())
        sel_columns = [
            TableColumn(
                field=_col,
                title=_col,
            )
            for _col in self.dfs["train"].columns
        ]
        self.sel_table = DataTable(source=sel_source, columns=sel_columns, **kwargs)

        def update_selection(selected_df):
            """
            Callback function.
            """
            # push results to bokeh data source
            sel_source.data = selected_df.to_dict(orient="list")

            self._good(
                f"Selection updater: latest selection with {selected_df.shape[0]} entries."
            )

        self._callback_update_selection = update_selection

    def df_deduplicate(self):
        """
        ???+ note "Cross-deduplicate data entries by feature between subsets."
        """
        self._info("Deduplicating...")
        # for data entry accounting
        before, after = dict(), dict()

        # deduplicating rule: entries that come LATER are of higher priority
        ordered_subsets = [
            *self.__class__.SCRATCH_SUBSETS,
            *self.__class__.PUBLIC_SUBSETS,
            *self.__class__.PRIVATE_SUBSETS,
        ]

        # keep track of which df has which columns and which rows came from which subset
        columns = dict()
        for _key in ordered_subsets:
            before[_key] = self.dfs[_key].shape[0]
            columns[_key] = self.dfs[_key].columns
            self.dfs[_key]["__subset"] = _key

        # concatenate in order and deduplicate
        overall_df = pd.concat(
            [self.dfs[_key] for _key in ordered_subsets], axis=0, sort=False
        )
        overall_df.drop_duplicates(
            subset=[self.__class__.FEATURE_KEY], keep="last", inplace=True
        )
        overall_df.reset_index(drop=True, inplace=True)

        # cut up slices
        for _key in ordered_subsets:
            self.dfs[_key] = overall_df[overall_df["__subset"] == _key].reset_index(
                drop=True, inplace=False
            )[columns[_key]]
            after[_key] = self.dfs[_key].shape[0]
            self._info(f"--subset {_key} rows: {before[_key]} -> {after[_key]}.")

    def synchronize_dictl_to_df(self):
        """
        ???+ note "Re-make dataframes from lists of dictionaries."
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
        ???+ note "Re-make lists of dictionaries from dataframes."
        """
        self.dictls = dict()
        for _key, _df in self.dfs.items():
            self.dictls[_key] = _df.to_dict(orient="records")

    def compute_2d_embedding(self, vectorizer, method, **kwargs):
        """
        ???+ note "Get embeddings in the xy-plane and return the dimensionality reducer."
            Reference: [`DimensionalityReducer`](https://github.com/phurwicz/hover/blob/main/hover/core/representation/reduction.py)

            | Param        | Type       | Description                        |
            | :----------- | :--------- | :--------------------------------- |
            | `vectorizer` | `callable` | the feature -> vector function     |
            | `method`     | `str`      | arg for `DimensionalityReducer`    |
            | `**kwargs`   |            | kwargs for `DimensionalityReducer` |
        """
        from hover.core.representation.reduction import DimensionalityReducer

        # prepare input vectors to manifold learning
        fit_subset = [*self.__class__.SCRATCH_SUBSETS, *self.__class__.PUBLIC_SUBSETS]
        trans_subset = [*self.__class__.PRIVATE_SUBSETS]

        assert not set(fit_subset).intersection(set(trans_subset)), "Unexpected overlap"

        # compute vectors and keep track which where to slice the array for fitting
        feature_inp = []
        for _key in fit_subset:
            feature_inp += self.dfs[_key][self.__class__.FEATURE_KEY].tolist()
        fit_num = len(feature_inp)
        for _key in trans_subset:
            feature_inp += self.dfs[_key][self.__class__.FEATURE_KEY].tolist()
        trans_arr = np.array([vectorizer(_inp) for _inp in tqdm(feature_inp)])

        # initialize and fit manifold learning reducer using specified subarray
        self._info(f"Fit-transforming {method.upper()} on {fit_num} samples...")
        reducer = DimensionalityReducer(trans_arr[:fit_num])
        fit_embedding = reducer.fit_transform(method, **kwargs)

        # compute embedding of the whole dataset
        self._info(
            f"Transforming {method.upper()} on {trans_arr.shape[0]-fit_num} samples..."
        )
        trans_embedding = reducer.transform(trans_arr[fit_num:], method)

        # assign x and y coordinates to dataset
        start_idx = 0
        for _subset, _embedding in [
            (fit_subset, fit_embedding),
            (trans_subset, trans_embedding),
        ]:
            # edge case: embedding is too small
            if _embedding.shape[0] < 1:
                for _key in _subset:
                    assert (
                        self.dfs[_key].shape[0] == 0
                    ), "Expected empty df due to empty embedding"
                continue
            for _key in _subset:
                _length = self.dfs[_key].shape[0]
                self.dfs[_key]["x"] = pd.Series(
                    _embedding[start_idx : (start_idx + _length), 0]
                )
                self.dfs[_key]["y"] = pd.Series(
                    _embedding[start_idx : (start_idx + _length), 1]
                )
                start_idx += _length

        return reducer

    def loader(self, key, vectorizer, batch_size=64, smoothing_coeff=0.0):
        """
        ???+ note "Prepare a torch `Dataloader` for training or evaluation."
            | Param        | Type       | Description                        |
            | :----------- | :--------- | :--------------------------------- |
            | `key`        | `str`      | subset of data, e.g. `"train"`     |
            | `vectorizer` | `callable` | the feature -> vector function     |
            | `batch_size` | `int`      | size per batch                     |
            | `smoothing_coeff` | `float` | portion of probability to equally split between classes |
        """
        # lazy import: missing torch should not break the rest of the class
        from hover.utils.torch_helper import vector_dataloader, one_hot, label_smoothing

        # take the slice that has a meaningful label
        df = self.dfs[key][self.dfs[key]["label"] != module_config.ABSTAIN_DECODED]

        # edge case: valid slice is too small
        if df.shape[0] < 1:
            raise ValueError(f"Subset {key} has too few samples ({df.shape[0]})")
        batch_size = min(batch_size, df.shape[0])

        labels = df["label"].apply(lambda x: self.label_encoder[x]).tolist()
        features = df[self.__class__.FEATURE_KEY].tolist()
        output_vectors = one_hot(labels, num_classes=len(self.classes))

        self._info(f"Preparing {key} input vectors...")
        input_vectors = [vectorizer(_f) for _f in tqdm(features)]
        if smoothing_coeff > 0.0:
            output_vectors = label_smoothing(
                output_vectors, coefficient=smoothing_coeff
            )
        self._info(f"Preparing {key} data loader...")
        loader = vector_dataloader(input_vectors, output_vectors, batch_size=batch_size)
        self._good(
            f"Prepared {key} loader consisting of {len(features)} examples with batch size {batch_size}"
        )
        return loader


class SupervisableTextDataset(SupervisableDataset):
    """
    ???+ note "Can add text-specific methods."
    """

    FEATURE_KEY = "text"
