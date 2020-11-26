"""Interactive explorers mostly based on Bokeh."""
import numpy as np
from wasabi import msg as logger
from bokeh.plotting import figure
from bokeh.models import CustomJS, ColumnDataSource, CDSView, IndexFilter
from bokeh.layouts import column, row
from abc import ABC, abstractmethod
from hover import module_config
from hover.utils.misc import current_time
from .local_config import bokeh_hover_tooltip


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

    DATA_KEYS = []

    def __init__(self, **kwargs):
        """
        Operations shared by all child classes.
        (1) settle the figure settings by using child class defaults + kwargs overrides
        (2) create a blank figure under such settings
        (3) create widgets that child classes can override
        """
        self.figure_settings = self.__class__.DEFAULT_FIGURE_KWARGS.copy()
        self.figure_settings.update(kwargs)
        self.reset_figure()
        self.reset_data()
        self.setup_widgets()

    def reset_figure(self):
        """Start over on the figure."""
        self.figure = figure(**self.figure_settings)

    def reset_data(self):
        """Start over on data that comprise the figure."""
        self.dfs = {_key: None for _key in self.__class__.DATA_KEYS}
        self.sources = {_key: None for _key in self.__class__.DATA_KEYS}

    def setup_widgets(self):
        """
        Prepare widgets for interactive functionality.

        Create positive/negative text search boxes.
        """
        from bokeh.models import TextInput

        self.search_pos = TextInput(
            title="Text contains (plain text, or /pattern/flag for regex):",
            width_policy="fit",
            height_policy="fit",
        )
        self.search_neg = TextInput(
            title="Text does not contain:", width_policy="fit", height_policy="fit"
        )

    def layout_widgets(self):
        """Define the layout of widgets."""
        return column(self.search_pos, self.search_neg)

    def view(self):
        """Define the layout of the whole explorer."""
        return column(self.layout_widgets(), self.figure)

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

    DATA_KEYS = ["raw"]

    BACKGROUND_GLYPH_KWARGS = {
        "color": "gainsboro",
        "line_alpha": 0.3,
        "legend_label": "unlabeled",
    }

    SEARCH_FLAG_TO_PARAMS = {
        "size": ("size", 10, 5, 7),
        "fill_alpha": ("fill_alpha", 0.6, 0.4, 0.5),
        "color": ("color", "coral", "linen", "gainsboro"),
    }

    def __init__(self, df_raw, **kwargs):
        """
        Requires the input dataframe to contain:

        (1) "x" and "y" columns for coordinates;
        (2) a "text" column for data point tooltips.
        """
        super().__init__(**kwargs)

        # prepare plot-ready dataframe for train set
        for _key in ["text", "x", "y"]:
            assert _key in df_raw.columns
        self.dfs["raw"] = df_raw.copy()

        # plot the train set as a background
        self.background_kwargs = self.__class__.BACKGROUND_GLYPH_KWARGS.copy()
        self.sources["raw"] = ColumnDataSource(self.dfs["raw"])
        self._activate_search_builtin()

    def _activate_search_builtin(self):
        """
        Highlight positive search results and mute negative search results.

        Note that this is a template method which heavily depend on class attributes.
        """
        for _flag, _params in self.__class__.SEARCH_FLAG_TO_PARAMS.items():
            logger.info(f"Activated {_flag} to respond to the search widgets.")
            _params = self.__class__.SEARCH_FLAG_TO_PARAMS[_flag]
            for _key in self.__class__.DATA_KEYS:
                self.background_kwargs = self.activate_search(
                    self.sources[_key], self.background_kwargs, altered_param=_params
                )

    def plot(self, *args, **kwargs):
        """
        (Re)-plot the corpus.
        Called just once per instance most of the time.
        """
        self.reset_figure()
        self.figure.circle(
            "x", "y", source=self.sources["raw"], **self.background_kwargs
        )


class BokehCorpusAnnotator(BokehForLabeledText):
    """
    [SERVER ONLY]
    Annoate text data points via callbacks.
    """

    DATA_KEYS = ["raw"]

    def __init__(self, df_raw, **kwargs):
        """
        Requires the input dataframe to contain:

        (1) "x" and "y" columns for coordinates;
        (2) a "text" column for data point tooltips.

        Optional:
        (3) a "label" column that defaults to ABSTAIN.
        """
        super().__init__(**kwargs)

        for _key in ["text", "x", "y"]:
            assert _key in df_raw.columns
        self.dfs["raw"] = df_raw.copy()
        if not "label" in self.dfs["raw"].columns:
            self.dfs["raw"]["label"] = module_config.ABSTAIN_DECODED

        self.sources["raw"] = ColumnDataSource(self.dfs["raw"])
        self.plot_kwargs = {"line_alpha": 0.3}
        self.reset_source()
        self.plot()

    def reset_source(self):
        self.sources["raw"].data = self.dfs["raw"]
        self.plot_kwargs = self.activate_search(
            self.sources["raw"], self.plot_kwargs, altered_param=("size", 10, 5, 7)
        )
        self.plot_kwargs = self.activate_search(
            self.sources["raw"],
            self.plot_kwargs,
            altered_param=("fill_alpha", 0.4, 0.05, 0.2),
        )

    def layout_widgets(self):
        """
        Define the layout of widgets.
        """
        first_row = row(self.search_pos, self.search_neg)
        second_row = row(
            self.annotator_input, self.annotator_apply, self.annotator_export
        )
        return column(first_row, second_row)

    def setup_widgets(self):
        """
        Create annotator widgets and assign Python callbacks.
        """
        from bokeh.models import TextInput, Button
        from bokeh.events import ButtonClick

        super().setup_widgets()

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

        def apply():
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

            self.reset_source()
            self.plot()
            logger.good(f"Updated annotator plot at {current_time()}")

        def export():
            """
            A callback on clicking the 'self.annotator_export' button.

            Saves the dataframe to a pickle.
            """
            from dill import dump
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"bokeh-annotated-df-{timestamp}.pkl"
            with open(filename, "wb") as f:
                dump(self.dfs["raw"], f)
            logger.good(f"Saved DataFrame to {filename}")

        self.annotator_apply.on_event(ButtonClick, apply)
        self.annotator_export.on_event(ButtonClick, export)

    def plot(self):
        """
        Re-plot with the new labels.

        Determines the label->color mapping dynamically.
        """
        from bokeh.transform import factor_cmap

        all_labels = sorted(set(self.dfs["raw"]["label"].values), reverse=True)
        assert len(all_labels) <= 20, "Too many labels to support"
        cmap = "Category10_10" if len(all_labels) <= 10 else "Category20_20"

        self.figure.circle(
            x="x",
            y="y",
            color=factor_cmap("label", cmap, all_labels),
            legend_field="label",
            source=self.sources["raw"],
            **self.plot_kwargs,
        )


class BokehMarginExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with two versions of labels.
    Could be useful for A/B tests.
    Currently not considering multi-label scenarios.
    """

    DATA_KEYS = ["raw"]

    BACKGROUND_GLYPH_KWARGS = {
        "color": "gainsboro",
        "line_alpha": 0.3,
        "fill_alpha": 0.0,
    }

    SEARCH_FLAG_TO_PARAMS = {"size": ("size", 10, 5, 7)}

    def __init__(self, df_raw, label_col_a, label_col_b, **kwargs):
        """
        On top of the requirements of the parent class,
        the input dataframe should contain:

        (1) label_col_a and label_col_b for "label margins".
        """
        super().__init__(df_raw, **kwargs)

        for _key in ["text", label_col_a, label_col_b, "x", "y"]:
            assert _key in self.dfs["raw"].columns
        self.label_col_a = label_col_a
        self.label_col_b = label_col_b

    def plot(self, label, **kwargs):
        """
        Plot the margins about a single label.
        """

        # prepare plot settings
        axes = ("x", "y")
        eff_kwargs = self.background_kwargs.copy()
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
            _marker(*axes, source=self.sources["raw"], view=_view, **eff_kwargs)


class BokehSnorkelExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with labeling function outputs.
    """

    DATA_KEYS = ["raw", "labeled"]

    BACKGROUND_GLYPH_KWARGS = {"line_alpha": 0.6, "fill_alpha": 0.0, "size": 7}

    SEARCH_FLAG_TO_PARAMS = {
        "size": ("size", 10, 5, 7),
        "fill_alpha": ("fill_alpha", 0.6, 0.0, 0.3),
    }

    def __init__(self, df_raw, df_labeled, **kwargs):
        """
        On top of the requirements of the parent class,
        the df_labeled input dataframe should contain:

        (1) a "label" column for "ground truths".
        """
        super().__init__(df_raw, **kwargs)

        # add 'label' column to df_raw
        if not "label" in self.dfs["raw"].columns:
            self.dfs["raw"]["label"] = module_config.ABSTAIN_DECODED

        # prepare plot-ready dataframe for df_labeled
        for _key in ["text", "label", "x", "y"]:
            assert _key in df_labeled.columns
        self.dfs["labeled"] = df_labeled.copy()
        self.sources["labeled"] = ColumnDataSource(self.dfs["labeled"])

        # initialize a list to keep track of plotted LFs
        self.lfs = []

    def plot(self, lf, L_raw=None, L_labeled=None, include=("C", "I", "M"), **kwargs):
        """
        Plot a single labeling function.

        :param lf: labeling function decorated by @hover.utils.snorkel_helper.labeling_function()
        :param L_raw: labeling function predictions, in decoded labels, on df_raw.
        :param L_labeled: labeling function predictions, in decoded labels, on df_labeled.
        :param include: subsets to show, which can be correct(C)/incorrect(I)/missed(M)/hit(H).
        """
        # keep track of added LF
        self.lfs.append(lf)

        # calculate predicted labels if not provided
        if L_raw is None:
            L_raw = self.dfs["raw"].apply(lf.row_to_label, axis=1).values
        if L_labeled is None:
            L_labeled = self.dfs["labeled"].apply(lf.row_to_label, axis=1).values

        # prepare plot settings
        axes = ("x", "y")
        decoded_targets = [lf.label_decoder[_target] for _target in lf.targets]
        eff_kwargs = self.background_kwargs.copy()
        eff_kwargs.update(kwargs)
        eff_kwargs["legend_label"] = f"{', '.join(decoded_targets)} | {lf.name}"

        # create correct/incorrect/missed/hit subsets
        to_plot = []
        if "C" in include:
            to_plot.append(
                {"view": self._view_correct(L_labeled), "marker": self.figure.square}
            )
        if "I" in include:
            to_plot.append(
                {"view": self._view_incorrect(L_labeled), "marker": self.figure.x}
            )
        if "M" in include:
            to_plot.append(
                {
                    "view": self._view_missed(L_labeled, lf.targets),
                    "marker": self.figure.cross,
                }
            )
        if "H" in include:
            to_plot.append(
                {"view": self._view_hit(L_raw), "marker": self.figure.circle}
            )

        # plot created subsets
        for _dict in to_plot:
            _view = _dict["view"]
            _marker = _dict["marker"]
            _marker(*axes, source=_view.source, view=_view, **eff_kwargs)

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
