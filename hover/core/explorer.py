"""Interactive explorers mostly based on Bokeh."""
import numpy as np
from wasabi import msg as logger
from bokeh.plotting import figure
from bokeh.models import CustomJS, ColumnDataSource, CDSView, IndexFilter
from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
from abc import ABC, abstractmethod
from hover import module_config
from hover.utils.misc import current_time
from .local_config import bokeh_hover_tooltip


def auto_cmap(labels):
    """
    Find an appropriate color map based on provide labels.
    """
    assert len(labels) <= 20, "Too many labels to support"
    cmap = "Category10_10" if len(labels) <= 10 else "Category20_20"
    return cmap


class BokehForLabeledText(ABC):
    """
    Base class that keeps template explorer settings.
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

    def __init__(self, df_dict, **kwargs):
        """
        Operations shared by all child classes.

        (1) settle the figure settings by using child class defaults + kwargs overrides
        (2) settle the glyph settings by using child class defaults
        (3) create widgets that child classes can override
        (4) create data sources the correspond to class-specific data subsets.
        (5) activate builtin search callbacks depending on the child class.
        (6) create a (likely) blank figure under such settings
        """
        logger.divider(f"Initializing {self.__class__.__name__}")
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

    def reset_figure(self):
        """Start over on the figure."""
        logger.info("Resetting figure")
        self.figure.renderers.clear()

    def _setup_widgets(self):
        """
        Prepare widgets for interactive functionality.

        Create positive/negative text search boxes.
        """
        from bokeh.models import TextInput, CheckboxButtonGroup

        # set up text search widgets, without assigning callbacks yet
        # to provide more flexibility with callbacks
        logger.info("Setting up widgets")
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
        logger.info("Setting up dfs")
        expected_keys = set(self.__class__.DATA_KEY_TO_KWARGS.keys())
        assert (
            set(df_dict.keys()) == expected_keys
        ), f"Expected the keys of df_dict to be exactly {expected_keys}"

        self.dfs = {
            _key: (_df.copy() if copy else _df) for _key, _df in df_dict.items()
        }

    def _setup_sources(self):
        """
        Create (NOT UPDATE) ColumnDataSource objects.

        Intended to be extended in child classes for pre/post processing.
        """
        logger.info("Setting up sources")
        self.sources = {_key: ColumnDataSource(_df) for _key, _df in self.dfs.items()}

    def _update_sources(self):
        """
        Update the sources with the corresponding dfs.

        Note that it seems mandatory to re-activate the search widgets.
        This is because the source loses plotting kwargs.
        """
        for _key in self.__class__.DATA_KEY_TO_KWARGS.keys():
            self.sources[_key].data = self.dfs[_key]
        self._activate_search_builtin()

    def _activate_search_builtin(self):
        """
        Typically called once during initialization.
        Highlight positive search results and mute negative search results.

        Note that this is a template method which heavily depends on class attributes.
        """
        logger.info("Activating built-in search")
        for _key, _dict in self.__class__.DATA_KEY_TO_KWARGS.items():
            for _flag, _params in _dict["search"].items():
                logger.info(
                    f"Activated {_flag} on subset {_key} to respond to the search widgets."
                )
                self.glyph_kwargs[_key] = self.activate_search(
                    self.sources[_key], self.glyph_kwargs[_key], altered_param=_params
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


class BokehCorpusExplorer(BokehForLabeledText):
    """
    Plot unlabeled, 2-D-vectorized text data points in a corpus.
    """

    DATA_KEY_TO_KWARGS = {
        "raw": {
            "constant": {"line_alpha": 0.4},
            "search": {
                "size": ("size", 10, 5, 7),
                "fill_alpha": ("fill_alpha", 0.4, 0.1, 0.2),
                "color": ("color", "coral", "linen", "gainsboro"),
            },
        }
    }

    def __init__(self, df_dict, **kwargs):
        """
        Requires the input dataframe to contain:

        (1) "x" and "y" columns for coordinates;
        (2) a "text" column for data point tooltips.
        """
        super().__init__(df_dict, **kwargs)

    def _setup_dfs(self, df_dict, **kwargs):
        """Extending from the parent method."""
        for _key in self.__class__.DATA_KEY_TO_KWARGS.keys():
            for _col in ["text", "x", "y"]:
                assert _col in df_dict[_key].columns

        super()._setup_dfs(df_dict, **kwargs)

    def plot(self, *args, **kwargs):
        """
        (Re)-plot the corpus.
        Called just once per instance most of the time.
        """
        self.figure.circle(
            "x", "y", name="raw", source=self.sources["raw"], **self.glyph_kwargs["raw"]
        )


class BokehCorpusAnnotator(BokehCorpusExplorer):
    """
    [SERVER ONLY]
    Annoate text data points via callbacks.
    """

    DATA_KEY_TO_KWARGS = {
        "raw": {
            "constant": {"line_alpha": 0.3},
            "search": {
                "size": ("size", 10, 5, 7),
                "fill_alpha": ("fill_alpha", 0.3, 0.05, 0.2),
            },
        }
    }

    def __init__(self, df_dict, **kwargs):
        """Conceptually the same as the parent method."""
        super().__init__(df_dict, **kwargs)

    def _setup_dfs(self, df_dict, **kwargs):
        """
        Extending from the parent method.

        Add a "label" column if it is not present.
        """
        super()._setup_dfs(df_dict, **kwargs)

        if not "label" in self.dfs["raw"].columns:
            self.dfs["raw"]["label"] = module_config.ABSTAIN_DECODED

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
        from bokeh.models import TextInput, Button

        super()._setup_widgets()

        self.annotator_input = TextInput(title="Label")
        self.annotator_apply = Button(
            label="Apply",
            button_type="primary",
            height_policy="fit",
            width_policy="min",
        )
        self.annotator_export = Button(
            label="Export",
            button_type="success",
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
                logger.warn("Did not select any data points.")
                return
            example_old = self.dfs["raw"].at[selected_idx[0], "label"]
            self.dfs["raw"].at[selected_idx, "label"] = label
            example_new = self.dfs["raw"].at[selected_idx[0], "label"]
            logger.good(f"Updated DataFrame, e.g. {example_old} -> {example_new}")

            self._update_sources()
            self.plot()
            logger.good(f"Updated annotator plot at {current_time()}")

        def callback_export(path=None):
            """
            A callback on clicking the 'self.annotator_export' button.

            Saves the dataframe to a pickle.
            """
            from dill import dump

            # auto-determine the export path
            if path is None:
                timestamp = current_time("%Y%m%d%H%M%S")
                path = f".bokeh-annotated-df-{timestamp}.pkl"

            # save a pickle, then send a message
            # note that excel/csv can be problematic with certain kinds of data
            with open(path, "wb") as f:
                dump(self.dfs["raw"], f)
            logger.good(f"Saved DataFrame to {path}")

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
        all_labels = sorted(set(self.dfs["raw"]["label"].values), reverse=True)
        cmap = auto_cmap(all_labels)

        self.figure.circle(
            x="x",
            y="y",
            name="raw",
            color=factor_cmap("label", cmap, all_labels),
            legend_field="label",
            source=self.sources["raw"],
            **self.glyph_kwargs["raw"],
        )


class BokehSoftLabelExplorer(BokehCorpusExplorer):
    """
    Plot text data points according to its label and confidence.
    Currently not considering multi-label scenarios.
    """

    DATA_KEY_TO_KWARGS = {
        "raw": {
            "constant": {"line_alpha": 0.5},
            "search": {"size": ("size", 10, 5, 7)},
        },
        "labeled": {
            "constant": {"line_alpha": 0.5},
            "search": {"size": ("size", 10, 5, 7)},
        },
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
                    label=False,
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
        for _key in self.__class__.DATA_KEY_TO_KWARGS.keys():
            for _col in [self.label_col, self.score_col]:
                assert _col in df_dict[_key].columns

        super()._setup_dfs(df_dict, **kwargs)

    def plot(self, **kwargs):
        """
        Plot the confidence map.
        """

        # auto-detect all labels
        all_labels = set()
        for _key in self.__class__.DATA_KEY_TO_KWARGS.keys():
            _df = self.dfs[_key]
            _labels = set(_df[self.label_col].values)
            all_labels = all_labels.union(_labels)
        all_labels = sorted(all_labels, reverse=True)
        cmap = auto_cmap(all_labels)

        for _key in self.__class__.DATA_KEY_TO_KWARGS.keys():
            # prepare plot settings
            preset_kwargs = {
                "legend_field": self.label_col,
                "color": factor_cmap(self.label_col, cmap, all_labels),
                "fill_alpha": self.score_col,
            }
            eff_kwargs = self.glyph_kwargs[_key].copy()
            eff_kwargs.update(preset_kwargs)
            eff_kwargs.update(kwargs)

            self.figure.circle(
                "x", "y", name=_key, source=self.sources[_key], **eff_kwargs
            )


class BokehMarginExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with two versions of labels.
    Could be useful for A/B tests.
    Currently not considering multi-label scenarios.
    """

    DATA_KEY_TO_KWARGS = {
        "raw": {
            "constant": {"color": "gainsboro", "line_alpha": 0.5, "fill_alpha": 0.0},
            "search": {"size": ("size", 10, 5, 7)},
        }
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

        # prepare plot settings
        axes = ("x", "y")
        eff_kwargs = self.glyph_kwargs["raw"].copy()
        eff_kwargs.update(kwargs)
        eff_kwargs["legend_label"] = f"{label}"

        # create agreement/increment/decrement subsets
        col_a_pos = np.where(self.dfs["raw"][self.label_col_a] == label)[0].tolist()
        col_a_neg = np.where(self.dfs["raw"][self.label_col_a] != label)[0].tolist()
        col_b_pos = np.where(self.dfs["raw"][self.label_col_b] == label)[0].tolist()
        col_b_neg = np.where(self.dfs["raw"][self.label_col_b] != label)[0].tolist()
        agreement_view = CDSView(
            source=self.sources["raw"],
            filters=[IndexFilter(col_a_pos), IndexFilter(col_b_pos)],
        )
        increment_view = CDSView(
            source=self.sources["raw"],
            filters=[IndexFilter(col_a_neg), IndexFilter(col_b_pos)],
        )
        decrement_view = CDSView(
            source=self.sources["raw"],
            filters=[IndexFilter(col_a_pos), IndexFilter(col_b_neg)],
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
            _marker(
                *axes, name="raw", source=self.sources["raw"], view=_view, **eff_kwargs
            )


class BokehSnorkelExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with labeling function outputs.
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

        assert "label" in self.dfs["labeled"].columns
        if not "label" in self.dfs["raw"].columns:
            self.dfs["raw"]["label"] = module_config.ABSTAIN_DECODED

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
        axes = ("x", "y")
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
            _view = _dict["view"]
            _marker = _dict["marker"]
            _kwargs = _dict["kwargs"]
            _marker(*axes, source=_view.source, view=_view, **_kwargs)

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
