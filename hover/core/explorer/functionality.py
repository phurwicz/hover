"""
???+ note "Intermediate classes based on the functionality."
"""
import numpy as np
from collections import OrderedDict
from bokeh.models import CDSView, IndexFilter, Dropdown, Button
from bokeh.palettes import Category20
from bokeh.layouts import row
from hover import module_config
from hover.utils.misc import current_time
from hover.utils.bokeh_helper import bokeh_hover_tooltip
from .local_config import SOURCE_COLOR_FIELD, SOURCE_ALPHA_FIELD, SEARCH_SCORE_FIELD
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

    def _setup_widgets(self):
        """
        ???+ note "Create score range slider that filters selections."
        """
        from bokeh.models import CheckboxGroup

        super()._setup_widgets()

        self.search_filter_box = CheckboxGroup(
            labels=["use as selection filter"], active=[]
        )

        def filter_flag():
            return bool(0 in self.search_filter_box.active)

        def filter_by_search(indices, subset):
            """
            Filter selection with search results on a subset.
            """
            search_scores = self.sources[subset].data[SEARCH_SCORE_FIELD]
            matched = set(np.where(np.array(search_scores) > 0)[0])
            return indices.intersection(matched)

        for _key in self.sources.keys():
            self._selection_filters[_key].data.add(
                lambda indices, subset: filter_by_search(indices, subset)
                if filter_flag()
                else indices
            )

        # when toggled as active, search changes trigger selection filter
        self.search_pos.on_change(
            "value",
            lambda attr, old, new: self._trigger_selection_filters()
            if filter_flag()
            else None,
        )
        self.search_neg.on_change(
            "value",
            lambda attr, old, new: self._trigger_selection_filters()
            if filter_flag()
            else None,
        )

        # active toggles always trigger selection filter
        self.search_filter_box.on_change(
            "active", lambda attr, old, new: self._trigger_selection_filters()
        )

    def plot(self):
        """
        ???+ note "Plot all data points."
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

            This is during initialization or re-plotting, creating a new attribute column for each data source.
        """
        color_dict = self.auto_color_mapping()

        for _key, _df in self.dfs.items():
            _color = (
                _df["label"]
                .apply(lambda label: color_dict.get(label, "gainsboro"))
                .tolist()
            )
            self.sources[_key].add(_color, SOURCE_COLOR_FIELD)

    def _update_colors(self):
        """
        ???+ note "Infer glyph colors from the label dynamically."

            This is during annotation callbacks, patching an existing column for the `raw` subset only.
        """
        # infer glyph colors dynamically
        color_dict = self.auto_color_mapping()

        color_list = (
            self.dfs["raw"]["label"]
            .apply(lambda label: color_dict.get(label, "gainsboro"))
            .tolist()
        )
        self.sources["raw"].patch(
            {SOURCE_COLOR_FIELD: [(slice(len(color_list)), color_list)]}
        )
        self._good(f"Updated annotator plot at {current_time()}")

    def _setup_widgets(self):
        """
        ???+ note "Create annotator widgets and assign Python callbacks."
        """
        from bokeh.models import TextInput

        super()._setup_widgets()

        self.annotator_input = TextInput(title="Label:")
        self.annotator_apply = Button(
            label="Apply",
            button_type="primary",
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
                    "attempting annotation: did not select any data points. Eligible subset is 'raw'."
                )
                return

            # update label in both the df and the data source
            self.dfs["raw"].loc[selected_idx, "label"] = label
            for _idx in selected_idx:
                _idx = int(_idx)
                self.sources["raw"].patch({"label": [(_idx, label)]})
            self._good(f"applied {len(selected_idx)} annotations: {label}")

            self._update_colors()

        # assign the callback and keep the reference
        self._callback_apply = callback_apply
        self.annotator_apply.on_click(self._callback_apply)
        self.annotator_apply.on_click(self._callback_subset_display)

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

    def _build_tooltip(self, extra):
        """
        ???+ note "On top of the parent method, add the soft label fields to the tooltip."
            | Param            | Type   | Description                  |
            | :--------------- | :----- | :--------------------------- |
            | `extra`          | `str`  | user-supplied extra HTML |
        """
        standard = bokeh_hover_tooltip(
            **self.__class__.TOOLTIP_KWARGS,
            custom={"Soft Label": self.label_col, "Soft Score": self.score_col},
        )
        return f"{standard}\n{extra}"

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

    def _setup_widgets(self):
        """
        ???+ note "Create score range slider that filters selections."
        """
        from bokeh.models import RangeSlider, CheckboxGroup

        super()._setup_widgets()

        self.score_range = RangeSlider(
            start=0.0,
            end=1.0,
            value=(0.0, 1.0),
            step=0.01,
            title="Score range",
        )
        self.score_filter_box = CheckboxGroup(
            labels=["use as selection filter"], active=[]
        )
        self.score_filter = row(self.score_range, self.score_filter_box)

        def filter_flag():
            return bool(0 in self.score_filter_box.active)

        def subroutine(df, lower, upper):
            """
            Calculate indices with score between lower/upper bounds.
            """
            keep_l = set(np.where(df[self.score_col] >= lower)[0])
            keep_u = set(np.where(df[self.score_col] <= upper)[0])
            kept = keep_l.intersection(keep_u)
            return kept

        def filter_by_score(indices, subset):
            """
            Filter selection with slider range on a subset.
            """
            in_range = subroutine(self.dfs[subset], *self.score_range.value)
            return indices.intersection(in_range)

        # selection change triggers score filter on the changed subset IFF filter box is toggled
        for _key in self.sources.keys():
            self._selection_filters[_key].data.add(
                lambda indices, subset: filter_by_score(indices, subset)
                if filter_flag()
                else indices
            )

        # when toggled as active, score range change triggers selection filter
        self.score_range.on_change(
            "value",
            lambda attr, old, new: self._trigger_selection_filters()
            if filter_flag()
            else None,
        )

        # active toggles always trigger selection filter
        self.score_filter_box.on_change(
            "active", lambda attr, old, new: self._trigger_selection_filters()
        )

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
        self.palette = list(Category20[20])
        self._subscribed_lf_list = None

    def _setup_sources(self):
        """
        ???+ note "Create data structures that source interactions will need."
        """
        # keep track of plotted LFs and glyphs, which will interact with sources
        self.lf_data = OrderedDict()
        super()._setup_sources()

    @property
    def subscribed_lf_list(self):
        """
        ???+ note "A list of LFs to which the explorer can be lazily synchronized."

            Intended for recipes where the user can modify LFs without having access to the explorer.
        """
        return self._subscribed_lf_list

    @subscribed_lf_list.setter
    def subscribed_lf_list(self, lf_list):
        """
        ???+ note "Subscribe to a list of LFs."
        """
        assert isinstance(lf_list, list), f"Expected a list of LFs, got {lf_list}"
        if self.subscribed_lf_list is None:
            self._good("Subscribed to a labeling function list BY REFERENCE.")
        else:
            self._warn("Changing labeling function list subscription.")
        self._subscribed_lf_list = lf_list
        self._callback_refresh_lf_menu()

    def _setup_widgets(self):
        """
        ???+ note "Create labeling function support widgets and assign Python callbacks."
        """
        super()._setup_widgets()
        self._subroutine_setup_lf_list_refresher()
        self._subroutine_setup_lf_apply_trigger()
        self._subroutine_setup_lf_filter_trigger()

    def _subroutine_setup_lf_list_refresher(self):
        """
        ???+ note "Create widget for refreshing LF list and replotting."
        """
        self.lf_list_refresher = Button(
            label="Refresh Functions",
            height_policy="fit",
            width_policy="min",
        )

        def callback_refresh_lf_plot():
            """
            Re-plot according to subscribed_lf_list.
            """
            if self.subscribed_lf_list is None:
                self._warn("cannot refresh LF plot without subscribed LF list.")
                return
            lf_names_to_keep = set([_lf.name for _lf in self.subscribed_lf_list])
            lf_names_to_drop = set(self.lf_data.keys()).difference(lf_names_to_keep)
            for _lf_name in lf_names_to_drop:
                self.unplot_lf(_lf_name)
            for _lf in self.subscribed_lf_list:
                self.plot_lf(_lf)

        def callback_refresh_lf_menu():
            """
            The menu was assigned by value and needs to stay consistent with LF updates.
            To be triggered in self.plot_new_lf() and self.unplot_lf().
            """
            self.lf_apply_trigger.menu = list(self.lf_data.keys())
            self.lf_filter_trigger.menu = list(self.lf_data.keys())

        self._callback_refresh_lf_menu = callback_refresh_lf_menu
        self.lf_list_refresher.on_click(callback_refresh_lf_plot)
        # self.lf_list_refresher.on_click(callback_refresh_lf_menu)

    def _subroutine_setup_lf_apply_trigger(self):
        """
        ???+ note "Create widget for applying LFs on data."
        """
        self.lf_apply_trigger = Dropdown(
            label="Apply Labels",
            button_type="warning",
            menu=list(self.lf_data.keys()),
            height_policy="fit",
            width_policy="min",
        )

        def callback_apply(event):
            """
            A callback on clicking the 'self.lf_apply_trigger' button.

            Update labels in the source similarly to the annotator.
            However, in this explorer, because LFs already use color, the produced labels will not.
            """
            lf = self.lf_data[event.item]["lf"]
            assert callable(lf), f"Expected a function, got {lf}"

            selected_idx = self.sources["raw"].selected.indices
            if not selected_idx:
                self._warn(
                    "attempting labeling by function: did not select any data points. Eligible subset is 'raw'."
                )
                return

            labels = self.dfs["raw"].iloc[selected_idx].apply(lf, axis=1).values
            num_nontrivial = len(
                list(filter(lambda l: l != module_config.ABSTAIN_DECODED, labels))
            )

            # update label in both the df and the data source
            self.dfs["raw"].loc[selected_idx, "label"] = labels
            for _idx, _label in zip(selected_idx, labels):
                _idx = int(_idx)
                self.sources["raw"].patch({"label": [(_idx, _label)]})
            self._info(
                f"applied {num_nontrivial}/{len(labels)} annotations by func {lf.name}"
            )

        self.lf_apply_trigger.on_click(callback_apply)

    def _subroutine_setup_lf_filter_trigger(self):
        """
        ???+ note "Create widget for using LFs to filter data."
        """
        self.lf_filter_trigger = Dropdown(
            label="Use as Selection Filter",
            button_type="primary",
            menu=list(self.lf_data.keys()),
            height_policy="fit",
            width_policy="min",
        )

        def callback_filter(event):
            """
            A callback on clicking the 'self.lf_filter_trigger' button.

            Update selected indices in a one-time manner.
            """
            lf = self.lf_data[event.item]["lf"]
            assert callable(lf), f"Expected a function, got {lf}"

            for _key, _source in self.sources.items():
                _selected = _source.selected.indices
                _labels = self.dfs[_key].iloc[_selected].apply(lf, axis=1).values
                _kept = [
                    _idx
                    for _idx, _label in zip(_selected, _labels)
                    if _label != module_config.ABSTAIN_DECODED
                ]
                self.sources[_key].selected.indices = _kept

        self.lf_filter_trigger.on_click(callback_filter)

    def _postprocess_sources(self):
        """
        ???+ note "Refresh all LF glyphs because data source has changed."
        """
        for _lf_name in self.lf_data.keys():
            self.refresh_glyphs(_lf_name)

    def plot(self, *args, **kwargs):
        """
        ???+ note "Plot the raw subset in the background."
        """
        self.figure.circle(
            "x", "y", name="raw", source=self.sources["raw"], **self.glyph_kwargs["raw"]
        )
        self._good(f"Plotted subset raw with {self.dfs['raw'].shape[0]} points")

    def plot_lf(self, lf, **kwargs):
        """
        ???+ note "Add or refresh a single labeling function on the plot."
            | Param       | Type             | Description                  |
            | :---------- | :--------------- | :--------------------------- |
            | `lf`        | `callable`       | labeling function decorated by `@labeling_function()` from `hover.utils.snorkel_helper` |
            | `**kwargs`  |             | forwarded to `self.plot_new_lf()` |
        """
        # keep track of added LF
        if lf.name in self.lf_data:
            # skip if the functions are identical
            if self.lf_data[lf.name]["lf"] is lf:
                return
            # overwrite the function and refresh glyphs
            self.lf_data[lf.name]["lf"] = lf
            self.refresh_glyphs(lf.name)
            return

        self.plot_new_lf(lf, **kwargs)

    def unplot_lf(self, lf_name):
        """
        ???+ note "Remove a single labeling function from the plot."
            | Param     | Type   | Description               |
            | :-------- | :----- | :------------------------ |
            | `lf_name` | `str`  | name of labeling function |
        """
        assert lf_name in self.lf_data, f"trying to remove non-existing LF: {lf_name}"

        data_dict = self.lf_data.pop(lf_name)
        lf, glyph_dict = data_dict["lf"], data_dict["glyphs"]
        assert lf.name == lf_name, f"LF name mismatch: {lf.name} vs {lf_name}"

        # remove from legend, checking that there is exactly one entry
        legend_idx_to_pop = None
        for i, _item in enumerate(self.figure.legend.items):
            _label = _item.label.get("value", "")
            if _label == lf_name:
                assert legend_idx_to_pop is None, f"Legend collision: {lf_name}"
                legend_idx_to_pop = i
        assert isinstance(legend_idx_to_pop, int), f"Missing from legend: {lf_name}"
        self.figure.legend.items.pop(legend_idx_to_pop)

        # remove from renderers
        # get indices to pop in ascending order
        renderer_indices_to_pop = []
        for i, _renderer in enumerate(self.figure.renderers):
            if lf_name in _renderer.glyph.tags:
                renderer_indices_to_pop.append(i)
        # check that the number of glyphs founded matches expected value
        num_fnd, num_exp = len(renderer_indices_to_pop), len(glyph_dict)
        assert num_fnd == num_exp, f"Glyph mismatch: {num_fnd} vs. {num_exp}"
        # process indices in descending order to avoid shifts
        for i in renderer_indices_to_pop[::-1]:
            self.figure.renderers.pop(i)

        # return color to palette so that another LF can use it
        self.palette.append(data_dict["color"])

        self._callback_refresh_lf_menu()
        self._good(f"Unplotted LF {lf_name}")

    def refresh_glyphs(self, lf_name):
        """
        ???+ note "Refresh the glyph(s) of a single LF based on its name."
            | Param     | Type   | Description               |
            | :-------- | :----- | :------------------------ |
            | `lf_name` | `str`  | name of labeling function |

            Assumes that specified C/I/M/H glyphs are stored.
            1. re-compute L_raw/L_labeled and CDSViews
            2. update the view for each glyph
        """
        assert lf_name in self.lf_data, f"trying to refresh non-existing LF: {lf_name}"

        lf = self.lf_data[lf_name]["lf"]
        L_raw = self.dfs["raw"].apply(lf, axis=1).values
        L_labeled = self.dfs["labeled"].apply(lf, axis=1).values

        glyph_codes = self.lf_data[lf_name]["glyphs"].keys()
        if "C" in glyph_codes:
            c_view = self._view_correct(L_labeled)
            self.lf_data[lf_name]["glyphs"]["C"].view = c_view
        if "I" in glyph_codes:
            i_view = self._view_incorrect(L_labeled)
            self.lf_data[lf_name]["glyphs"]["I"].view = i_view
        if "M" in glyph_codes:
            m_view = self._view_missed(L_labeled, lf.targets)
            self.lf_data[lf_name]["glyphs"]["M"].view = m_view
        if "H" in glyph_codes:
            h_view = self._view_hit(L_raw)
            self.lf_data[lf_name]["glyphs"]["H"].view = h_view

        self._good(f"Refreshed the glyphs of LF {lf_name}")

    def plot_new_lf(
        self, lf, L_raw=None, L_labeled=None, include=("C", "I", "M"), **kwargs
    ):
        """
        ???+ note "Plot a single labeling function and keep its settings for update."
            | Param       | Type             | Description                  |
            | :---------- | :--------------- | :--------------------------- |
            | `lf`        | `callable`       | labeling function decorated by `@labeling_function()` from `hover.utils.snorkel_helper` |
            | `L_raw`     | `np.ndarray`     | predictions, in decoded `str`, on the `"raw"` set |
            | `L_labeled` | `np.ndarray`     | predictions, in decoded `str`, on the `"labeled"` set |
            | `include`   | `tuple` of `str` | "C" for correct, "I" for incorrect, "M" for missed", "H" for hit: types of predictions to make visible in the plot |
            | `**kwargs`  |                  | forwarded to plotting markers |


            - lf: labeling function decorated by `@labeling_function()` from `hover.utils.snorkel_helper`
            - L_raw: numpy.ndarray
            - L_labeled: numpy.ndarray
            - include: subsets to show, which can be correct(C)/incorrect(I)/missed(M)/hit(H).
        """
        # existing LF should not trigger this method
        assert lf.name not in self.lf_data, f"LF collision: {lf.name}"

        # calculate predicted labels if not provided
        if L_raw is None:
            L_raw = self.dfs["raw"].apply(lf, axis=1).values
        if L_labeled is None:
            L_labeled = self.dfs["labeled"].apply(lf, axis=1).values

        # prepare plot settings
        assert self.palette, f"Palette depleted, # LFs: {len(self.lf_data)}"
        legend_label = lf.name
        color = self.palette.pop(0)

        raw_glyph_kwargs = self.glyph_kwargs["raw"].copy()
        raw_glyph_kwargs["legend_label"] = legend_label
        raw_glyph_kwargs["color"] = color
        raw_glyph_kwargs.update(kwargs)

        labeled_glyph_kwargs = self.glyph_kwargs["labeled"].copy()
        labeled_glyph_kwargs["legend_label"] = legend_label
        labeled_glyph_kwargs["color"] = color
        labeled_glyph_kwargs.update(kwargs)

        # create dictionary to prepare for dynamic lf & glyph updates
        data_dict = {"lf": lf, "color": color, "glyphs": {}}

        # add correct/incorrect/missed/hit glyphs
        if "C" in include:
            view = self._view_correct(L_labeled)
            data_dict["glyphs"]["C"] = self.figure.square(
                "x",
                "y",
                source=view.source,
                view=view,
                name="labeled",
                tags=[lf.name],
                **labeled_glyph_kwargs,
            )
        if "I" in include:
            view = self._view_incorrect(L_labeled)
            data_dict["glyphs"]["I"] = self.figure.x(
                "x",
                "y",
                source=view.source,
                view=view,
                name="labeled",
                tags=[lf.name],
                **labeled_glyph_kwargs,
            )
        if "M" in include:
            view = self._view_missed(L_labeled, lf.targets)
            data_dict["glyphs"]["M"] = self.figure.cross(
                "x",
                "y",
                source=view.source,
                view=view,
                name="labeled",
                tags=[lf.name],
                **labeled_glyph_kwargs,
            )
        if "H" in include:
            view = self._view_hit(L_raw)
            data_dict["glyphs"]["H"] = self.figure.circle(
                "x",
                "y",
                source=view.source,
                view=view,
                name="raw",
                tags=[lf.name],
                **raw_glyph_kwargs,
            )

        # assign the completed dictionary
        self.lf_data[lf.name] = data_dict
        # reflect LF update in widgets
        self._callback_refresh_lf_menu()

        self._good(f"Plotted new LF {lf.name}")

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
