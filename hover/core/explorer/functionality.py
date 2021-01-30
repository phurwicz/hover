"""
???+ note "Intermediate classes based on the functionality."
"""
import numpy as np
from bokeh.models import CDSView, IndexFilter
from bokeh.palettes import Category20
from hover import module_config
from hover.utils.misc import current_time
from .local_config import bokeh_hover_tooltip, SOURCE_COLOR_FIELD, SOURCE_ALPHA_FIELD
from .base import BokehBaseExplorer


class BokehDataFinder(BokehBaseExplorer):
    """
    ???+ note "Plot data points in grey ('gainsboro') and highlight search positives in coral."

        Features:

        -   the search widgets will highlight the results through a change of color, which gives arguably the best focus.
    """

    SUBSET_GLYPH_KWARGS = {
        _key: {
            "constant": {"line_alpha": 0.4},
            "search": {
                "size": ("size", 10, 5, 7),
                "fill_alpha": ("fill_alpha", 0.4, 0.1, 0.2),
                "color": ("color", "coral", "linen", "gainsboro"),
            },
        }
        for _key in ["raw", "train", "dev", "test"]
    }

    def plot(self, *args, **kwargs):
        """
        ???+ note "Plot all data points."
            | Param      | Type  | Description           |
            | :--------- | :---- | :-------------------- |
            | `*args`    |       | for consistency       |
            | `**kwargs` |       | for consistency       |
        """
        for _key, _source in self.sources.items():
            self.figure.circle(
                "x", "y", name=_key, source=_source, **self.glyph_kwargs[_key]
            )
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")


class BokehDataAnnotator(BokehBaseExplorer):
    """
    ???+ note "Annoate data points via callbacks on the buttons."

        Features:

        - alter values in the 'label' column through the widgets.
    """

    SUBSET_GLYPH_KWARGS = {
        _key: {
            "constant": {"line_alpha": 0.3},
            "search": {
                "size": ("size", 10, 5, 7),
                "fill_alpha": ("fill_alpha", 0.5, 0.1, 0.4),
            },
        }
        for _key in ["raw", "train", "dev", "test"]
    }

    def _postprocess_sources(self):
        """
        ???+ note "Infer glyph colors from the label dynamically."
        """
        # infer glyph color from labels
        color_dict = self.auto_color_mapping()

        def get_color(label):
            return color_dict.get(label, "gainsboro")

        for _key, _df in self.dfs.items():
            _color = _df["label"].apply(get_color).tolist()
            self.sources[_key].add(_color, SOURCE_COLOR_FIELD)

    def _setup_widgets(self):
        """
        ???+ note "Create annotator widgets and assign Python callbacks."
        """
        from bokeh.models import TextInput, Button, Dropdown

        super()._setup_widgets()

        self.annotator_input = TextInput(title="Label:")
        self.annotator_apply = Button(
            label="Apply",
            button_type="primary",
            height_policy="fit",
            width_policy="min",
        )
        self.annotator_export = Dropdown(
            label="Export",
            button_type="warning",
            menu=["Excel", "CSV", "JSON", "pickle"],
            height_policy="fit",
            width_policy="min",
        )

        def callback_apply():
            """
            A callback on clicking the 'self.annotator_apply' button.

            Update labels in the source.
            """
            label = self.annotator_input.value
            selected_idx = self.sources["raw"].selected.indices
            if not selected_idx:
                self._warn(
                    "Attempting annotation: did not select any data points. Eligible subset is 'raw'."
                )
                return
            example_old = self.dfs["raw"].at[selected_idx[0], "label"]
            self.dfs["raw"].at[selected_idx, "label"] = label
            example_new = self.dfs["raw"].at[selected_idx[0], "label"]
            self._good(
                f"Applied {len(selected_idx)} annotations: {label} (e.g. {example_old} -> {example_new})"
            )

            self._update_sources()
            self._good(f"Updated annotator plot at {current_time()}")

        def callback_export(event, path_root=None):
            """
            A callback on clicking the 'self.annotator_export' button.

            Saves the dataframe to a pickle.
            """
            import pandas as pd

            export_format = event.item

            # auto-determine the export path root
            if path_root is None:
                timestamp = current_time("%Y%m%d%H%M%S")
                path_root = f"hover-annotated-df-{timestamp}"

            export_df = pd.concat(self.dfs, axis=0, sort=False, ignore_index=True)

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

        # keep the references to the callbacks
        self._callback_apply = callback_apply
        self._callback_export = callback_export

        # assign callbacks
        self.annotator_apply.on_click(self._callback_apply)
        self.annotator_apply.on_click(self._callback_subset_display)
        self.annotator_export.on_click(self._callback_export)

    def plot(self):
        """
        ???+ note "Re-plot all data points with the new labels."
            Overrides the parent method.
            Determines the label -> color mapping dynamically.
        """
        for _key, _source in self.sources.items():
            self.figure.circle(
                "x",
                "y",
                name=_key,
                color=SOURCE_COLOR_FIELD,
                source=_source,
                **self.glyph_kwargs[_key],
            )
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")


class BokehSoftLabelExplorer(BokehBaseExplorer):
    """
    ???+ note "Plot data points according to their labels and confidence scores."

        Features:

        - the predicted label will correspond to fill_color.
        - the confidence score, assumed to be a float between 0.0 and 1.0, will be reflected through fill_alpha.
        - currently not considering multi-label scenarios.
    """

    SUBSET_GLYPH_KWARGS = {
        _key: {"constant": {"line_alpha": 0.5}, "search": {"size": ("size", 10, 5, 7)}}
        for _key in ["raw", "train", "dev"]
    }

    def __init__(self, df_dict, label_col, score_col, **kwargs):
        """
        ???+ note "Additional construtor"
            On top of the requirements of the parent class,
            the input dataframe should contain:

            - label_col and score_col for "soft predictions".

            | Param       | Type   | Description                  |
            | :---------- | :----- | :--------------------------- |
            | `df_dict`   | `dict` | `str` -> `DataFrame` mapping |
            | `label_col` | `str`  | column for the soft label    |
            | `score_col` | `str`  | column for the soft score    |
            | `**kwargs`  |        | forwarded to `bokeh.plotting.figure` |
        """
        assert label_col != "label", "'label' field is reserved"
        self.label_col = label_col
        self.score_col = score_col
        super().__init__(df_dict, **kwargs)

    def _build_tooltip(self):
        """
        ???+ note "On top of the parent method, add the soft label fields to the tooltip."
        """
        return bokeh_hover_tooltip(
            **self.__class__.TOOLTIP_KWARGS,
            custom={"Soft Label": self.label_col, "Soft Score": self.score_col},
        )

    def _setup_dfs(self, df_dict, **kwargs):
        """
        ???+ note "On top of the parent method, add filler values to additional columns."
            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `df_dict`  | `dict` | `str` -> `DataFrame` mapping |
            | `**kwargs` |        | forwarded to the parent method |
        """
        super()._setup_dfs(df_dict, **kwargs)

        for _key, _df in self.dfs.items():
            if self.label_col not in _df.columns:
                _df[self.label_col] = module_config.ABSTAIN_DECODED
            if self.score_col not in _df.columns:
                _df[self.score_col] = 0.5

    def _postprocess_sources(self):
        """
        ???+ note "Infer glyph colors from the label dynamically."
        """
        # infer glyph color from labels
        color_dict = self.auto_color_mapping()

        def get_color(label):
            return color_dict.get(label, "gainsboro")

        # infer glyph alpha from pseudo-percentile of soft label scores
        scores = np.concatenate(
            [_df[self.score_col].tolist() for _df in self.dfs.values()]
        )
        scores_mean = scores.mean()
        scores_std = scores.std() + 1e-4

        def pseudo_percentile(confidence, lower=0.1, upper=0.9):
            # pretend that 2*std on each side covers everything
            unit_shift = upper - lower / 4
            # shift = unit_shift * z_score
            shift = unit_shift * (confidence - scores_mean) / scores_std
            percentile = 0.5 + shift
            return min(upper, max(lower, percentile))

        # infer alpha from score percentiles
        for _key, _df in self.dfs.items():
            _color = _df[self.label_col].apply(get_color).tolist()
            _alpha = _df[self.score_col].apply(pseudo_percentile).tolist()
            self.sources[_key].add(_color, SOURCE_COLOR_FIELD)
            self.sources[_key].add(_alpha, SOURCE_ALPHA_FIELD)

    def plot(self, **kwargs):
        """
        ???+ note "Plot all data points, setting color alpha based on the soft score."
            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `**kwargs` |        | forwarded to plotting markers |
        """
        for _key, _source in self.sources.items():
            # prepare plot settings
            preset_kwargs = {
                "color": SOURCE_COLOR_FIELD,
                "fill_alpha": SOURCE_ALPHA_FIELD,
            }
            eff_kwargs = self.glyph_kwargs[_key].copy()
            eff_kwargs.update(preset_kwargs)
            eff_kwargs.update(kwargs)

            self.figure.circle("x", "y", name=_key, source=_source, **eff_kwargs)
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")


class BokehMarginExplorer(BokehBaseExplorer):
    """
    ???+ note "Plot data points along with two versions of labels."
        Could be useful for A/B tests.

        Features:

        - can choose to only plot the margins about specific labels.
        - currently not considering multi-label scenarios.
    """

    SUBSET_GLYPH_KWARGS = {
        _key: {
            "constant": {"color": "gainsboro", "line_alpha": 0.5, "fill_alpha": 0.0},
            "search": {"size": ("size", 10, 5, 7)},
        }
        for _key in ["raw", "train", "dev"]
    }

    def __init__(self, df_dict, label_col_a, label_col_b, **kwargs):
        """
        ???+ note "Additional construtor"
            On top of the requirements of the parent class,
            the input dataframe should contain:

            - label_col_a and label_col_b for "label margins".

            | Param         | Type   | Description                  |
            | :------------ | :----- | :--------------------------- |
            | `df_dict`     | `dict` | `str` -> `DataFrame` mapping |
            | `label_col_a` | `str`  | column for label set A       |
            | `label_col_b` | `str`  | column for label set B       |
            | `**kwargs`    |        | forwarded to `bokeh.plotting.figure` |
        """
        self.label_col_a = label_col_a
        self.label_col_b = label_col_b
        super().__init__(df_dict, **kwargs)

    def _setup_dfs(self, df_dict, **kwargs):
        """
        ???+ note "On top of the parent method, add column checks."
            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `df_dict`  | `dict` | `str` -> `DataFrame` mapping |
            | `**kwargs` |        | forwarded to the parent method |
        """
        super()._setup_dfs(df_dict, **kwargs)

        for _key, _df in self.dfs.items():
            for _col in [self.label_col_a, self.label_col_b]:
                assert (
                    _col in _df.columns
                ), f"Expected column {_col} among {list(_df.columns)}"

    def plot(self, label, **kwargs):
        """
        ???+ note "Plot the margins about a single label."
            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `label`    |        | the label to plot about      |
            | `**kwargs` |        | forwarded to plotting markers |
        """

        for _key, _source in self.sources.items():
            # prepare plot settings
            eff_kwargs = self.glyph_kwargs[_key].copy()
            eff_kwargs.update(kwargs)
            eff_kwargs["legend_label"] = f"{label}"

            # create agreement/increment/decrement subsets
            col_a_pos = np.where(self.dfs[_key][self.label_col_a] == label)[0].tolist()
            col_a_neg = np.where(self.dfs[_key][self.label_col_a] != label)[0].tolist()
            col_b_pos = np.where(self.dfs[_key][self.label_col_b] == label)[0].tolist()
            col_b_neg = np.where(self.dfs[_key][self.label_col_b] != label)[0].tolist()
            agreement_view = CDSView(
                source=_source, filters=[IndexFilter(col_a_pos), IndexFilter(col_b_pos)]
            )
            increment_view = CDSView(
                source=_source, filters=[IndexFilter(col_a_neg), IndexFilter(col_b_pos)]
            )
            decrement_view = CDSView(
                source=_source, filters=[IndexFilter(col_a_pos), IndexFilter(col_b_neg)]
            )

            to_plot = [
                {"view": agreement_view, "marker": self.figure.square},
                {"view": increment_view, "marker": self.figure.x},
                {"view": decrement_view, "marker": self.figure.cross},
            ]

            # plot created subsets
            for _dict in to_plot:
                _view = _dict["view"]
                _marker = _dict["marker"]
                _marker("x", "y", name=_key, source=_source, view=_view, **eff_kwargs)


class BokehSnorkelExplorer(BokehBaseExplorer):
    """
    ???+ note "Plot data points along with labeling function (LF) outputs."

        Features:

        -   each labeling function corresponds to its own line_color.
        -   uses a different marker for each type of predictions: square for 'correct', x for 'incorrect', cross for 'missed', circle for 'hit'.
          -   'correct': the LF made a correct prediction on a point in the 'labeled' set.
          -   'incorrect': the LF made an incorrect prediction on a point in the 'labeled' set.
          -   'missed': the LF is capable of predicting the target class, but did not make such prediction on the particular point.
          -   'hit': the LF made a prediction on a point in the 'raw' set.
    """

    SUBSET_GLYPH_KWARGS = {
        "raw": {
            "constant": {"line_alpha": 1.0, "color": "gainsboro"},
            "search": {
                "size": ("size", 10, 5, 7),
                "fill_alpha": ("fill_alpha", 0.4, 0.05, 0.2),
            },
        },
        "labeled": {
            "constant": {"line_alpha": 1.0, "fill_alpha": 0.0},
            "search": {"size": ("size", 10, 5, 7)},
        },
    }

    def __init__(self, df_dict, **kwargs):
        """
        ???+ note "Additional construtor"
            Set up

            -   a list to keep track of plotted labeling functions.
            -   a palette for plotting labeling function predictions.

            | Param       | Type   | Description                  |
            | :---------- | :----- | :--------------------------- |
            | `df_dict`   | `dict` | `str` -> `DataFrame` mapping |
            | `**kwargs`  |        | forwarded to `bokeh.plotting.figure` |
        """
        super().__init__(df_dict, **kwargs)

        # initialize a list to keep track of plotted LFs
        self.lfs = []
        self.palette = Category20[20]

    def plot(self, *args, **kwargs):
        """
        ???+ note "Plot the raw subset in the background."
        """
        self.figure.circle(
            "x", "y", name="raw", source=self.sources["raw"], **self.glyph_kwargs["raw"]
        )
        self._good(f"Plotted subset raw with {self.dfs['raw'].shape[0]} points")

    def plot_lf(
        self, lf, L_raw=None, L_labeled=None, include=("C", "I", "M"), **kwargs
    ):
        """
        ???+ note "Plot about a single labeling function."
            | Param       | Type             | Description                  |
            | :---------- | :--------------- | :--------------------------- |
            | `lf`        | `callable`       | labeling function decorated by `@labeling_function()` from `hover.utils.snorkel_helper` |
            | `L_raw`     | `np.ndarray`     | predictions, in decoded `str`, on the `"raw"` set |
            | `L_labeled` | `np.ndarray`     | predictions, in decoded `str`, on the `"labeled"` set |
            | `include`   | `tuple` of `str` | "C" for correct, "I" for incorrect, "M" for missed", "H" for hit: types of predictions to make visible in the plot |
            | `**kwargs`  |                  | forwarded to plotting markers |


            - lf: labeling function decorated by `@labeling_function()` from `hover.utils.snorkel_helper`
            - L_raw: .
            - L_labeled: .
            - include: subsets to show, which can be correct(C)/incorrect(I)/missed(M)/hit(H).
        """
        # keep track of added LF
        self.lfs.append(lf)

        # calculate predicted labels if not provided
        if L_raw is None:
            L_raw = self.dfs["raw"].apply(lf, axis=1).values
        if L_labeled is None:
            L_labeled = self.dfs["labeled"].apply(lf, axis=1).values

        # prepare plot settings
        legend_label = f"{', '.join(lf.targets)} | {lf.name}"
        color = self.palette[len(self.lfs) - 1]

        raw_glyph_kwargs = self.glyph_kwargs["raw"].copy()
        raw_glyph_kwargs["legend_label"] = legend_label
        raw_glyph_kwargs["color"] = color
        raw_glyph_kwargs.update(kwargs)

        labeled_glyph_kwargs = self.glyph_kwargs["labeled"].copy()
        labeled_glyph_kwargs["legend_label"] = legend_label
        labeled_glyph_kwargs["color"] = color
        labeled_glyph_kwargs.update(kwargs)

        # create correct/incorrect/missed/hit subsets
        to_plot = []
        if "C" in include:
            to_plot.append(
                {
                    "name": "labeled",
                    "view": self._view_correct(L_labeled),
                    "marker": self.figure.square,
                    "kwargs": labeled_glyph_kwargs,
                }
            )
        if "I" in include:
            to_plot.append(
                {
                    "name": "labeled",
                    "view": self._view_incorrect(L_labeled),
                    "marker": self.figure.x,
                    "kwargs": labeled_glyph_kwargs,
                }
            )
        if "M" in include:
            to_plot.append(
                {
                    "name": "labeled",
                    "view": self._view_missed(L_labeled, lf.targets),
                    "marker": self.figure.cross,
                    "kwargs": labeled_glyph_kwargs,
                }
            )
        if "H" in include:
            to_plot.append(
                {
                    "name": "raw",
                    "view": self._view_hit(L_raw),
                    "marker": self.figure.circle,
                    "kwargs": raw_glyph_kwargs,
                }
            )

        # plot created subsets
        for _dict in to_plot:
            _name = _dict["name"]
            _view = _dict["view"]
            _marker = _dict["marker"]
            _kwargs = _dict["kwargs"]
            _marker("x", "y", source=_view.source, view=_view, name=_name, **_kwargs)

    def _view_correct(self, L_labeled):
        """
        ???+ note "Determine the portion correctly labeled by a labeling function."
            | Param       | Type         | Description                  |
            | :---------- | :----------- | :--------------------------- |
            | `L_labeled` | `np.ndarray` | predictions on the labeled subset |
        """
        agreed = self.dfs["labeled"]["label"].values == L_labeled
        attempted = L_labeled != module_config.ABSTAIN_DECODED
        indices = np.where(np.multiply(agreed, attempted))[0].tolist()
        view = CDSView(source=self.sources["labeled"], filters=[IndexFilter(indices)])
        return view

    def _view_incorrect(self, L_labeled):
        """
        ???+ note "Determine the portion incorrectly labeled by a labeling function."
            | Param       | Type         | Description                  |
            | :---------- | :----------- | :--------------------------- |
            | `L_labeled` | `np.ndarray` | predictions on the labeled subset |
        """
        disagreed = self.dfs["labeled"]["label"].values != L_labeled
        attempted = L_labeled != module_config.ABSTAIN_DECODED
        indices = np.where(np.multiply(disagreed, attempted))[0].tolist()
        view = CDSView(source=self.sources["labeled"], filters=[IndexFilter(indices)])
        return view

    def _view_missed(self, L_labeled, targets):
        """
        ???+ note "Determine the portion missed by a labeling function."
            | Param       | Type          | Description                  |
            | :---------- | :------------ | :--------------------------- |
            | `L_labeled` | `np.ndarray`  | predictions on the labeled subset |
            | `targets` | `list` of `str` | labels that the function aims for |
        """
        targetable = np.isin(self.dfs["labeled"]["label"], targets)
        abstained = L_labeled == module_config.ABSTAIN_DECODED
        indices = np.where(np.multiply(targetable, abstained))[0].tolist()
        view = CDSView(source=self.sources["labeled"], filters=[IndexFilter(indices)])
        return view

    def _view_hit(self, L_raw):
        """
        ???+ note "Determine the portion hit by a labeling function."
            | Param       | Type         | Description                  |
            | :---------- | :----------- | :--------------------------- |
            | `L_raw`     | `np.ndarray` | predictions on the raw subset |
        """
        indices = np.where(L_raw != module_config.ABSTAIN_DECODED)[0].tolist()
        view = CDSView(source=self.sources["raw"], filters=[IndexFilter(indices)])
        return view
