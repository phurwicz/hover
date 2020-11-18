"""
Interactive explorers mostly based on Bokeh.
"""
import pandas as pd
import numpy as np
import bokeh
import wrappy
from wasabi import msg as logger
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.layouts import column, row
from abc import ABC, abstractmethod
from copy import deepcopy
from hover import module_config
from .local_config import bokeh_hover_tooltip


class BokehForLabeledText(ABC):
    """
    Base class that keeps template figure settings.
    """

    def __init__(self, **kwargs):
        self.figure_settings = {
            "tools": [
                "pan",
                "wheel_zoom",
                "lasso_select",
                "box_select",
                "hover",
                "reset",
            ],
            "tooltips": bokeh_hover_tooltip(
                label=True, text=True, image=False, coords=True, index=True
            ),
            "output_backend": "webgl",
        }
        self.figure_settings.update(kwargs)
        self.reset_figure()
        self.setup_widgets()

    def reset_figure(self):
        from bokeh.plotting import figure

        self.figure = figure(**self.figure_settings)

    def setup_widgets(self):
        """
        Prepare widgets for interactive functionality.
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
        """
        Define the layout of widgets.
        """
        return column(self.search_pos, self.search_neg)

    def view(self):
        """
        Return a formatted figure for display.
        """

        layout = column(self.layout_widgets(), self.figure)
        return layout

    def activate_search(self, source, kwargs, altered_param=("size", 8, 3, 5)):
        """
        Enables string/regex search-and-highlight mechanism.
        Modifies plotting source and kwargs in-place.
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

    BACKGROUND_KWARGS = {
        "color": "gainsboro",
        "line_alpha": 0.3,
        "legend_label": "unlabeled",
    }

    def __init__(self, df_raw, **kwargs):
        super().__init__(**kwargs)

        # prepare plot-ready dataframe for train set
        for _key in ["text", "x", "y"]:
            assert _key in df_raw.columns
        self.df_raw = df_raw.copy()

        # plot the train set as a background
        self.background_kwargs = deepcopy(self.__class__.BACKGROUND_KWARGS)
        self.source = ColumnDataSource(self.df_raw)
        self._activate_search_on_corpus()

        self.figure.circle("x", "y", source=self.source, **self.background_kwargs)

    def _activate_search_on_corpus(self):
        """
        Assuming that there will not be a labeled dev set.
        """
        self.background_kwargs = self.activate_search(
            self.source, self.background_kwargs, altered_param=("size", 8, 3, 5)
        )
        self.background_kwargs = self.activate_search(
            self.source,
            self.background_kwargs,
            altered_param=("fill_alpha", 0.6, 0.4, 0.5),
        )
        self.background_kwargs = self.activate_search(
            self.source,
            self.background_kwargs,
            altered_param=("color", "coral", "linen", "gainsboro"),
        )

    def plot(self, *args, **kwargs):
        """
        Does nothing.
        """
        pass


class BokehCorpusAnnotator(BokehForLabeledText):
    """
    [SERVER ONLY]
    Annoate text data points via callbacks.
    """

    def __init__(self, df_working, **kwargs):

        super().__init__(**kwargs)

        for _key in ["text", "x", "y"]:
            assert _key in df_working.columns
        self.df_working = df_working.copy()
        if not "label" in self.df_working.columns:
            self.df_working["label"] = module_config.ABSTAIN_DECODED

        self.source = ColumnDataSource(self.df_working)
        self.plot_kwargs = {"line_alpha": 0.3}
        self.reset_source()
        self.plot()

    def reset_source(self):
        self.source.data = self.df_working
        self.plot_kwargs = self.activate_search(
            self.source, self.plot_kwargs, altered_param=("size", 8, 3, 5)
        )
        self.plot_kwargs = self.activate_search(
            self.source, self.plot_kwargs, altered_param=("fill_alpha", 0.4, 0.05, 0.2)
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
            # a callback on clicking self.annotator_apply
            # update labels in the source
            label = self.annotator_input.value
            selected_idx = self.source.selected.indices
            if not selected_idx:
                logger.warn("Did not select any data points.")
                return
            example_old = self.df_working.at[selected_idx[0], "label"]
            self.df_working.at[selected_idx, "label"] = label
            example_new = self.df_working.at[selected_idx[0], "label"]
            logger.good(f"Updated DataFrame, e.g. {example_old} -> {example_new}")

            self.reset_source()
            self.plot()
            logger.good(f"Updated annotator plot")

        def export():
            # a callback on clicking self.annotator_export
            import dill
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"bokeh-annotated-df-{timestamp}.pkl"
            with open(filename, "wb") as f:
                dill.dump(self.df_working, f)
            logger.good(f"Saved DataFrame to {filename}")

        self.annotator_apply.on_event(ButtonClick, apply)
        self.annotator_export.on_event(ButtonClick, export)

    def plot(self):
        """
        Re-plot with the new labels.
        """
        from bokeh.transform import factor_cmap

        all_labels = sorted(set(self.df_working["label"].values), reverse=True)
        assert len(all_labels) <= 20, "Too many labels to support"
        cmap = "Category10_10" if len(all_labels) <= 10 else "Category20_20"

        self.figure.circle(
            x="x",
            y="y",
            color=factor_cmap("label", cmap, all_labels),
            legend_field="label",
            source=self.source,
            **self.plot_kwargs,
        )


class BokehMarginExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with two versions of labels.
    Could be useful for A/B tests.
    Currently not considering multi-label scenarios.
    """

    BACKGROUND_KWARGS = {"color": "gainsboro", "line_alpha": 0.3, "fill_alpha": 0.0}

    def __init__(self, df_raw, label_col_a, label_col_b, **kwargs):
        super().__init__(df_raw, **kwargs)

        for _key in ["text", label_col_a, label_col_b, "x", "y"]:
            assert _key in self.df_raw.columns
        self.label_col_a = label_col_a
        self.label_col_b = label_col_b

    def _activate_search_on_corpus(self):
        """
        Overriding the parent method, because there will be labels.
        """
        self.background_kwargs = self.activate_search(
            self.source, self.background_kwargs, altered_param=("size", 6, 1, 3)
        )
        # self.background_kwargs = self.activate_search(
        #    self.source,
        #    self.background_kwargs,
        #    altered_param=("fill_alpha", 0.6, 0.0, 0.3),
        # )

    def plot(self, label, **kwargs):
        """
        Plot the margins about a single label.
        """
        from bokeh.models import CDSView, IndexFilter

        # prepare plot settings
        axes = ("x", "y")
        eff_kwargs = deepcopy(self.background_kwargs)
        eff_kwargs.update(kwargs)
        eff_kwargs["legend_label"] = f"{label}"

        # create agreement/increment/decrement subsets
        col_a_pos = np.where(self.df_raw[self.label_col_a] == label)[0].tolist()
        col_a_neg = np.where(self.df_raw[self.label_col_a] != label)[0].tolist()
        col_b_pos = np.where(self.df_raw[self.label_col_b] == label)[0].tolist()
        col_b_neg = np.where(self.df_raw[self.label_col_b] != label)[0].tolist()
        agreement_view = CDSView(
            source=self.source, filters=[IndexFilter(col_a_pos), IndexFilter(col_b_pos)]
        )
        increment_view = CDSView(
            source=self.source, filters=[IndexFilter(col_a_neg), IndexFilter(col_b_pos)]
        )
        decrement_view = CDSView(
            source=self.source, filters=[IndexFilter(col_a_pos), IndexFilter(col_b_neg)]
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
            _marker(*axes, source=self.source, view=_view, **eff_kwargs)


class BokehSnorkelExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with labeling function outputs.
    """

    def __init__(self, df_raw, df_labeled, **kwargs):
        super().__init__(df_raw, **kwargs)

        # add 'label' column to df_raw
        self.df_raw["label"] = "unlabeled"

        # prepare plot-ready dataframe for dev set
        for _key in ["text", "label", "x", "y"]:
            assert _key in df_labeled.columns
        self.df_labeled = df_labeled.copy()

        # initialize a list of LFs to enable toggles
        self.lfs = []

    def _activate_search_on_corpus(self, source, background_kwargs):
        """
        Overriding the parent method, because there will be a labeled dev set.
        """
        self.activate_search(source, background_kwargs, altered_param=("size", 6, 1, 3))
        self.activate_search(
            source, background_kwargs, altered_param=("fill_alpha", 0.6, 0.0, 0.3)
        )

    # def plot(self, lf, L_train, L_dev, include, **kwargs):
    def plot(self, lf, L_raw=None, L_labeled=None, include=["C", "I", "M"], **kwargs):
        """
        Plot a single labeling function.
        """
        # keep track of added LF
        self.lfs.append(lf)

        # calculate predicted labels if not provided
        if L_raw is None:
            L_raw = self.df_raw.apply(lf.row_to_label, axis=1).values
        if L_labeled is None:
            L_labeled = self.df_labeled.apply(lf.row_to_label, axis=1).values

        # prepare plot settings
        axes = ("x", "y")
        decoded_targets = [lf.label_decoder[_target] for _target in lf.targets]
        legend = f"{', '.join(decoded_targets)} | {lf.name}"
        template_kwargs = {"line_alpha": 0.6, "fill_alpha": 0.0, "size": 5}
        template_kwargs.update(kwargs)

        # create correct/incorrect/missed/hit subsets
        to_plot = []
        if "C" in include:
            to_plot.append(
                {
                    "source": self._source_correct(L_labeled),
                    "marker": self.figure.square,
                }
            )
        if "I" in include:
            to_plot.append(
                {"source": self._source_incorrect(L_labeled), "marker": self.figure.x}
            )
        if "M" in include:
            to_plot.append(
                {
                    "source": self._source_missed(L_labeled, lf.targets),
                    "marker": self.figure.cross,
                }
            )
        if "H" in include:
            to_plot.append(
                {"source": self._source_hit(L_raw), "marker": self.figure.circle}
            )

        # plot created subsets
        for _dict in to_plot:
            _source = _dict["source"]
            _marker = _dict["marker"]
            _kwargs = deepcopy(template_kwargs)
            self.activate_search(_source, _kwargs)
            _marker(*axes, source=_source, legend_label=legend, **_kwargs)

    @wrappy.todo("Review whether it's appropriate to create a ColumnDataSource")
    def _source_correct(self, L_labeled):
        """
        Determine the subset correctly labeled by a labeling function.
        """
        indices = self.df_labeled["label"].values == L_labeled
        return ColumnDataSource(self.df_labeled.iloc[indices])

    @wrappy.todo("Review whether it's appropriate to create a ColumnDataSource")
    def _source_incorrect(self, L_labeled):
        """
        Determine the subset incorrectly labeled by a labeling function.
        """
        disagreed = self.df_labeled["label"].values != L_labeled
        attempted = L_labeled != module_config.ABSTAIN_ENCODED
        indices = np.multiply(disagreed, attempted)
        return ColumnDataSource(self.df_labeled.iloc[indices])

    @wrappy.todo("Review whether it's appropriate to create a ColumnDataSource")
    def _source_missed(self, L_labeled, targets):
        """
        Determine the subset missed by a labeling function.
        """
        targetable = np.isin(self.df_labeled["label"], targets)
        abstained = L_labeled == module_config.ABSTAIN_DECODED
        indices = np.multiply(targetable, abstained)
        return ColumnDataSource(self.df_labeled.iloc[indices])

    @wrappy.todo("Review whether it's appropriate to create a ColumnDataSource")
    def _source_hit(self, L_raw):
        """
        Determine the subset hit by a labeling function.
        """
        indices = L_raw != module_config.ABSTAIN_DECODED
        return ColumnDataSource(self.df_raw.iloc[indices])
