"""Interactive explorers mostly based on Bokeh."""
import numpy as np
import pandas as pd
from collections import OrderedDict
from bokeh.plotting import figure
from bokeh.models import CustomJS, ColumnDataSource, CDSView, IndexFilter
from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
from abc import ABC, abstractmethod
from hover import module_config
from hover.core import Loggable
from hover.utils.misc import current_time
from .local_config import bokeh_hover_tooltip


class BokehForLabeledText(Loggable, ABC):
    """
    Base class that keeps template explorer settings.

    Assumes:

    - in supplied dataframes
      - (always) text data in a `text` column
      - (always) xy coordinates in `x` and `y` columns
      - (always) an index for the rows
      - (likely) classification label in a `label` column

    Does not assume:

    - what the explorer serves to do.
    """

    DEFAULT_FIGURE_KWARGS = {
        "tools": [
            # change the scope
            "pan",
            "wheel_zoom",
            # make selections
            "tap",
            "poly_select",
            "lasso_select",
            # make inspections
            "hover",
            # navigate changes
            "undo",
            "redo",
        ],
        # inspection details
        "tooltips": bokeh_hover_tooltip(
            label=True, text=True, image=False, coords=True, index=True
        ),
        # bokeh recommends webgl for scalability
        "output_backend": "webgl",
    }

    DATA_KEY_TO_KWARGS = {}

    MANDATORY_COLUMNS = ["text", "label", "x", "y"]

    def __init__(self, df_dict, **kwargs):
        """
        Operations shared by all child classes.

        - settle the figure settings by using child class defaults & kwargs overrides
        - settle the glyph settings by using child class defaults
        - create widgets that child classes can override
        - create data sources the correspond to class-specific data subsets.
        - activate builtin search callbacks depending on the child class.
        - create a (typically) blank figure under such settings
        """
        self.figure_kwargs = self.__class__.DEFAULT_FIGURE_KWARGS.copy()
        self.figure_kwargs.update(kwargs)
        self.glyph_kwargs = {
            _key: _dict["constant"].copy()
            for _key, _dict in self.__class__.DATA_KEY_TO_KWARGS.items()
        }
        self._setup_widgets()
        self._setup_dfs(df_dict)
        self._setup_sources()
        self._activate_search_builtin()
        self.figure = figure(**self.figure_kwargs)
        self.reset_figure()

    @classmethod
    def from_dataset(cls, dataset, subset_mapping, *args, **kwargs):
        """
        Construct from a SupervisableDataset.
        """
        # local import to avoid import cycles
        from hover.core.dataset import SupervisableDataset

        assert isinstance(dataset, SupervisableDataset)
        df_dict = {_v: dataset.dfs[_k] for _k, _v in subset_mapping.items()}
        return cls(df_dict, *args, **kwargs)

    def reset_figure(self):
        """Start over on the figure."""
        self._info("Resetting figure")
        self.figure.renderers.clear()

    def _setup_widgets(self):
        """
        Prepare widgets for interactive functionality.

        Create positive/negative text search boxes.
        """
        from bokeh.models import TextInput, CheckboxButtonGroup

        # set up text search widgets, without assigning callbacks yet
        # to provide more flexibility with callbacks
        self._info("Setting up widgets")
        self.search_pos = TextInput(
            title="Text contains (plain text, or /pattern/flag for regex):",
            width_policy="fit",
            height_policy="fit",
        )
        self.search_neg = TextInput(
            title="Text does not contain:", width_policy="fit", height_policy="fit"
        )

        # set up subset display toggles which do have clearly defined callbacks
        data_keys = list(self.__class__.DATA_KEY_TO_KWARGS.keys())
        self.data_key_button_group = CheckboxButtonGroup(
            labels=data_keys, active=list(range(len(data_keys)))
        )

        def update_data_key_display(active):
            visible_keys = {self.data_key_button_group.labels[idx] for idx in active}
            for _renderer in self.figure.renderers:
                # if the renderer has a name "on the list", update its visibility
                if _renderer.name in self.__class__.DATA_KEY_TO_KWARGS.keys():
                    _renderer.visible = _renderer.name in visible_keys

        # store the callback (useful, for example, during automated tests) and link it
        self.update_data_key_display = update_data_key_display
        self.data_key_button_group.on_click(self.update_data_key_display)

    def _layout_widgets(self):
        """Define the layout of widgets."""
        return column(self.search_pos, self.search_neg, self.data_key_button_group)

    def view(self):
        """Define the layout of the whole explorer."""
        return column(self._layout_widgets(), self.figure)

    def _setup_dfs(self, df_dict, copy=False):
        """
        Check and store DataFrames BY REFERENCE BY DEFAULT.

        Intended to be extended in child classes for pre/post processing.
        """
        self._info("Setting up DataFrames")
        supplied_keys = set(df_dict.keys())
        expected_keys = set(self.__class__.DATA_KEY_TO_KWARGS.keys())

        # perform high-level df key checks
        supplied_not_expected = supplied_keys.difference(expected_keys)
        expected_not_supplied = expected_keys.difference(supplied_keys)

        for _key in supplied_not_expected:
            self._warn(
                f"{self.__class__.__name__}.__init__(): got unexpected df key {_key}"
            )
        for _key in expected_not_supplied:
            self._warn(
                f"{self.__class__.__name__}.__init__(): missing expected df key {_key}"
            )

        # create df with column checks
        self.dfs = dict()
        for _key, _df in df_dict.items():
            if _key in expected_keys:
                for _col in self.__class__.MANDATORY_COLUMNS:
                    if not _col in _df.columns:
                        # edge case: DataFrame has zero rows
                        assert (
                            _df.shape[0] == 0
                        ), f"Missing column '{_col}' from non-empty {_key} DataFrame: found {list(_df.columns)}"
                        _df[_col] = None

                self.dfs[_key] = _df.copy() if copy else _df

    def _setup_sources(self):
        """
        Create (NOT UPDATE) ColumnDataSource objects.

        Intended to be extended in child classes for pre/post processing.
        """
        self._info("Setting up sources")
        self.sources = {_key: ColumnDataSource(_df) for _key, _df in self.dfs.items()}

    def _update_sources(self):
        """
        Update the sources with the corresponding dfs.

        Note that it seems mandatory to re-activate the search widgets.
        This is because the source loses plotting kwargs.
        """
        for _key in self.dfs.keys():
            self.sources[_key].data = self.dfs[_key]
        self._activate_search_builtin(verbose=False)

    def _activate_search_builtin(self, verbose=True):
        """
        Typically called once during initialization.
        Highlight positive search results and mute negative search results.

        Note that this is a template method which heavily depends on class attributes.
        """
        for _key, _dict in self.__class__.DATA_KEY_TO_KWARGS.items():
            if _key in self.sources.keys():
                _responding = list(_dict["search"].keys())
                for _flag, _params in _dict["search"].items():
                    self.glyph_kwargs[_key] = self.activate_search(
                        self.sources[_key],
                        self.glyph_kwargs[_key],
                        altered_param=_params,
                    )
                if verbose:
                    self._info(
                        f"Activated {_responding} on subset {_key} to respond to the search widgets."
                    )

    def activate_search(self, source, kwargs, altered_param=("size", 10, 5, 7)):
        """
        Enables string/regex search-and-highlight mechanism.

        Modifies the plotting source in-place.
        """
        assert isinstance(source, ColumnDataSource)
        assert isinstance(kwargs, dict)
        updated_kwargs = kwargs.copy()

        param_key, param_pos, param_neg, param_default = altered_param
        num_points = len(source.data["text"])
        default_param_list = [param_default] * num_points
        source.add(default_param_list, f"{param_key}")

        updated_kwargs[param_key] = param_key

        search_callback = CustomJS(
            args={
                "source": source,
                "key_pos": self.search_pos,
                "key_neg": self.search_neg,
                "param_pos": param_pos,
                "param_neg": param_neg,
                "param_default": param_default,
            },
            code=f"""
            const data = source.data;
            const text = data['text'];
            var arr = data['{param_key}'];
            """
            + """
            var search_pos = key_pos.value;
            var search_neg = key_neg.value;
            var valid_pos = (search_pos.length > 0);
            var valid_neg = (search_neg.length > 0);

            function determineAttr(candidate)
            {
                var score = 0;
                if (valid_pos) {
                    if (candidate.search(search_pos) >= 0) {
                        score += 1;
                    } else {
                        score -= 2;
                    }
                };
                if (valid_neg) {
                    if (candidate.search(search_neg) < 0) {
                        score += 1;
                    } else {
                        score -= 2;
                    }
                };
                if (score > 0) {
                    return param_pos;
                } else if (score < 0) {
                    return param_neg;
                } else {return param_default;}
            }

            function toRegex(search_key) {
                var match = search_key.match(new RegExp('^/(.*?)/([gimy]*)$'));
                if (match) {
                    return new RegExp(match[1], match[2]);
                } else {
                    return search_key;
                }
            }

            if (valid_pos) {search_pos = toRegex(search_pos);}
            if (valid_neg) {search_neg = toRegex(search_neg);}
            for (var i = 0; i < arr.length; i++) {
                arr[i] = determineAttr(text[i]);
            }

            source.change.emit()
            """,
        )

        self.search_pos.js_on_change("value", search_callback)
        self.search_neg.js_on_change("value", search_callback)
        return updated_kwargs

    def _prelink_check(self, other):
        """
        Sanity check before linking two explorers.
        """
        assert other is not self, "Self-loops are fordidden"
        assert isinstance(other, BokehForLabeledText), "Must link to BokehForLabelText"

    def link_selection(self, key, other, other_key):
        """
        Sync the selected indices between specified sources.
        """
        self._prelink_check(other)
        # link selection in a bidirectional manner
        sl, sr = self.sources[key], other.sources[other_key]
        sl.selected.js_link("indices", sr.selected, "indices")
        sr.selected.js_link("indices", sl.selected, "indices")

    def link_xy_range(self, other):
        """
        Sync plotting ranges on the xy-plane.
        """
        self._prelink_check(other)
        # link coordinate ranges in a bidirectional manner
        for _attr in ["start", "end"]:
            self.figure.x_range.js_link(_attr, other.figure.x_range, _attr)
            self.figure.y_range.js_link(_attr, other.figure.y_range, _attr)
            other.figure.x_range.js_link(_attr, self.figure.x_range, _attr)
            other.figure.y_range.js_link(_attr, self.figure.y_range, _attr)

    @abstractmethod
    def plot(self, *args, **kwargs):
        """
        Plot something onto the figure.
        """
        pass

    def auto_labels_cmap(self):
        """
        Find all labels and an appropriate color map.
        """
        labels = set()
        for _key in self.dfs.keys():
            labels = labels.union(set(self.dfs[_key]["label"].values))
        labels.discard(module_config.ABSTAIN_DECODED)
        labels = sorted(labels, reverse=True)

        assert len(labels) <= 20, "Too many labels to support (max at 20)"
        cmap = "Category10_10" if len(labels) <= 10 else "Category20_20"
        return labels, cmap

    def auto_legend_correction(self):
        """
        Find legend items and deduplicate by label.
        """
        if not hasattr(self.figure, "legend"):
            self._fail(f"Attempting auto_legend_correction when there is no legend")
            return
        # extract all items and start over
        items = self.figure.legend.items[:]
        self.figure.legend.items.clear()

        # use one item to hold all renderers matching its label
        label_to_item = OrderedDict()

        for _item in items:
            _label = _item.label.get("value", "")
            if not _label in label_to_item.keys():
                label_to_item[_label] = _item
            else:
                label_to_item[_label].renderers.extend(_item.renderers)

        # assign deduplicated items back to the legend
        self.figure.legend.items = list(label_to_item.values())
        return


class BokehCorpusExplorer(BokehForLabeledText):
    """
    Plot unlabeled, 2D-vectorized text data points in a corpus.

    Features:

    - the search widgets will highlight the results through a change of color, which is arguably the best visual effect.
    """

    DATA_KEY_TO_KWARGS = {
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

    def __init__(self, df_dict, **kwargs):
        """
        Requires the input dataframe to contain:

        (1) "x" and "y" columns for coordinates;
        (2) a "text" column for data point tooltips.
        """
        super().__init__(df_dict, **kwargs)

    def plot(self, *args, **kwargs):
        """
        (Re)-plot the corpus.
        Called just once per instance most of the time.
        """
        for _key, _source in self.sources.items():
            self.figure.circle(
                "x", "y", name=_key, source=_source, **self.glyph_kwargs[_key]
            )
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")


class BokehCorpusAnnotator(BokehCorpusExplorer):
    """
    Annoate text data points via callbacks.

    Features:

    - alter values in the 'label' column through the widgets.
    - **SERVER ONLY**: only works in a setting that allows Python callbacks.
    """

    DATA_KEY_TO_KWARGS = {
        _key: {
            "constant": {"line_alpha": 0.3},
            "search": {
                "size": ("size", 10, 5, 7),
                "fill_alpha": ("fill_alpha", 0.3, 0.1, 0.2),
            },
        }
        for _key in ["raw", "train", "dev", "test"]
    }

    def __init__(self, df_dict, **kwargs):
        """Conceptually the same as the parent method."""
        super().__init__(df_dict, **kwargs)

    def _layout_widgets(self):
        """Define the layout of widgets."""
        layout_rows = (
            row(self.search_pos, self.search_neg),
            row(self.data_key_button_group),
            row(self.annotator_input, self.annotator_apply, self.annotator_export),
        )
        return column(*layout_rows)

    def _setup_widgets(self):
        """
        Create annotator widgets and assign Python callbacks.
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
            self._good(f"Applied {len(selected_idx)} annotations: {example_new}")

            self._update_sources()
            self.plot()
            self._good(f"Updated annotator plot at {current_time()}")

        def callback_export(event, path_root=None):
            """
            A callback on clicking the 'self.annotator_export' button.

            Saves the dataframe to a pickle.
            """
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
        self.annotator_export.on_click(self._callback_export)

    def plot(self):
        """
        Re-plot with the new labels.

        Overrides the parent method.
        Determines the label->color mapping dynamically.
        """
        labels, cmap = self.auto_labels_cmap()

        for _key, _source in self.sources.items():
            self.figure.circle(
                "x",
                "y",
                name=_key,
                color=factor_cmap("label", cmap, labels),
                legend_group="label",
                source=_source,
                **self.glyph_kwargs[_key],
            )
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")

        self.auto_legend_correction()


class BokehSoftLabelExplorer(BokehCorpusExplorer):
    """
    Plot text data points according to their labels and confidence scores.

    Features:

    - the predicted label will correspond to fill_color.
    - the confidence score, assumed to be a float between 0.0 and 1.0, will be reflected through fill_alpha.
    - currently not considering multi-label scenarios.
    """

    DATA_KEY_TO_KWARGS = {
        _key: {"constant": {"line_alpha": 0.5}, "search": {"size": ("size", 10, 5, 7)}}
        for _key in ["raw", "train", "dev"]
    }

    def __init__(self, df_dict, label_col, score_col, **kwargs):
        """
        On top of the requirements of the parent class,
        the input dataframe should contain:

        (1) label_col and score_col for "soft predictions".
        """
        assert label_col != "label", "'label' field is reserved"
        self.label_col = label_col
        self.score_col = score_col
        kwargs.update(
            {
                "tooltips": bokeh_hover_tooltip(
                    label=True,
                    text=True,
                    image=False,
                    coords=True,
                    index=True,
                    custom={"Soft Label": self.label_col, "Soft Score": self.score_col},
                )
            }
        )
        super().__init__(df_dict, **kwargs)

    def _setup_dfs(self, df_dict, **kwargs):
        """Extending from the parent method."""
        super()._setup_dfs(df_dict, **kwargs)

        for _key, _df in self.dfs.items():
            if not self.label_col in _df.columns:
                _df[self.label_col] = module_config.ABSTAIN_DECODED
            if not self.score_col in _df.columns:
                _df[self.score_col] = 0.5

    def plot(self, **kwargs):
        """
        Plot the confidence map.
        """
        labels, cmap = self.auto_labels_cmap()

        for _key, _source in self.sources.items():
            # prepare plot settings
            preset_kwargs = {
                "legend_group": self.label_col,
                "color": factor_cmap(self.label_col, cmap, labels),
                "fill_alpha": self.score_col,
            }
            eff_kwargs = self.glyph_kwargs[_key].copy()
            eff_kwargs.update(preset_kwargs)
            eff_kwargs.update(kwargs)

            self.figure.circle("x", "y", name=_key, source=_source, **eff_kwargs)
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")

        self.auto_legend_correction()


class BokehMarginExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with two versions of labels.
    Could be useful for A/B tests.

    Features:

    - can choose to only plot the margins about specific labels.
    - currently not considering multi-label scenarios.
    """

    DATA_KEY_TO_KWARGS = {
        _key: {
            "constant": {"color": "gainsboro", "line_alpha": 0.5, "fill_alpha": 0.0},
            "search": {"size": ("size", 10, 5, 7)},
        }
        for _key in ["raw", "train", "dev"]
    }

    def __init__(self, df_dict, label_col_a, label_col_b, **kwargs):
        """
        On top of the requirements of the parent class,
        the input dataframe should contain:

        (1) label_col_a and label_col_b for "label margins".
        """
        self.label_col_a = label_col_a
        self.label_col_b = label_col_b
        super().__init__(df_dict, **kwargs)

    def _setup_dfs(self, df_dict, **kwargs):
        """Extending from the parent method."""
        for _key in [self.label_col_a, self.label_col_b]:
            assert _key in df_dict["raw"].columns

        super()._setup_dfs(df_dict, **kwargs)

    def plot(self, label, **kwargs):
        """
        Plot the margins about a single label.
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


class BokehSnorkelExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with labeling function (LF) outputs.

    Features:

    - each labeling function corresponds to its own line_color.
    - uses a different marker for each type of predictions: square for 'correct', x for 'incorrect', cross for 'missed', circle for 'hit'.
      - 'correct': the LF made a correct prediction on a point in the 'labeled' set.
      - 'incorrect': the LF made an incorrect prediction on a point in the 'labeled' set.
      - 'missed': the LF is capable of predicting the target class, but did not make such prediction on the particular point.
      - 'hit': the LF made a prediction on a point in the 'raw' set.
    """

    DATA_KEY_TO_KWARGS = {
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
        On top of the requirements of the parent class,
        the df_labeled input dataframe should contain:

        (1) a "label" column for "ground truths".
        """
        super().__init__(df_dict, **kwargs)

        # initialize a list to keep track of plotted LFs
        self.lfs = []
        self.palette = Category20[20]

    def _setup_dfs(self, df_dict, **kwargs):
        """Extending from the parent method."""
        super()._setup_dfs(df_dict, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Overriding the parent method.

        Plot only the raw subset.
        """
        self.figure.circle(
            "x", "y", name="raw", source=self.sources["raw"], **self.glyph_kwargs["raw"]
        )
        self._good(f"Plotted subset raw with {self.dfs['raw'].shape[0]} points")

    def plot_lf(
        self, lf, L_raw=None, L_labeled=None, include=("C", "I", "M"), **kwargs
    ):
        """
        Plot a single labeling function.

        - param lf: labeling function decorated by @hover.utils.snorkel_helper.labeling_function()
        - param L_raw: labeling function predictions, in decoded labels, on the raw df.
        - param L_labeled: labeling function predictions, in decoded labels, on the labeled df.
        - param include: subsets to show, which can be correct(C)/incorrect(I)/missed(M)/hit(H).
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
        Determine the subset correctly labeled by a labeling function.
        """
        agreed = self.dfs["labeled"]["label"].values == L_labeled
        attempted = L_labeled != module_config.ABSTAIN_DECODED
        indices = np.where(np.multiply(agreed, attempted))[0].tolist()
        view = CDSView(source=self.sources["labeled"], filters=[IndexFilter(indices)])
        return view

    def _view_incorrect(self, L_labeled):
        """
        Determine the subset incorrectly labeled by a labeling function.
        """
        disagreed = self.dfs["labeled"]["label"].values != L_labeled
        attempted = L_labeled != module_config.ABSTAIN_DECODED
        indices = np.where(np.multiply(disagreed, attempted))[0].tolist()
        view = CDSView(source=self.sources["labeled"], filters=[IndexFilter(indices)])
        return view

    def _view_missed(self, L_labeled, targets):
        """
        Determine the subset missed by a labeling function.
        """
        targetable = np.isin(self.dfs["labeled"]["label"], targets)
        abstained = L_labeled == module_config.ABSTAIN_DECODED
        indices = np.where(np.multiply(targetable, abstained))[0].tolist()
        view = CDSView(source=self.sources["labeled"], filters=[IndexFilter(indices)])
        return view

    def _view_hit(self, L_raw):
        """
        Determine the subset hit by a labeling function.
        """
        indices = np.where(L_raw != module_config.ABSTAIN_DECODED)[0].tolist()
        view = CDSView(source=self.sources["raw"], filters=[IndexFilter(indices)])
        return view
