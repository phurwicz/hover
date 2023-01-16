"""
???+ note "Base class(es) for ALL explorer implementations."
"""
import pandas as pd
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from bokeh.events import SelectionGeometry
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from hover.core import Loggable
from hover.core.local_config import (
    is_embedding_field,
    blank_callback_on_change as blank,
)
from hover.utils.bokeh_helper import bokeh_hover_tooltip
from hover.utils.meta.traceback import RichTracebackABCMeta
from hover.utils.misc import RootUnionFind
from .local_config import SEARCH_SCORE_FIELD

STANDARD_PLOT_TOOLS = [
    # change the scope
    "pan",
    "wheel_zoom",
    # make selections
    "tap",
    "poly_select",
    "lasso_select",
    # make inspections
    "hover",
]


class BokehBaseExplorer(Loggable, ABC, metaclass=RichTracebackABCMeta):
    """
    ???+ note "Base class for visually exploring data with `Bokeh`."
        Assumes:

        - in supplied dataframes
          - (always) xy coordinates in `x` and `y` columns;
          - (always) an index for the rows;
          - (always) classification label (or ABSTAIN) in a `label` column.

        Does not assume:

        - a specific form of data;
        - what the map serves to do.
    """

    SUBSET_GLYPH_KWARGS = {}
    DEFAULT_SUBSET_MAPPING = {_k: _k for _k in ["raw", "train", "dev", "test"]}
    SELECTION_PROCESSING_STAGES = ["save", "load", "write", "read"]

    PRIMARY_FEATURE = None
    MANDATORY_COLUMNS = ["label"]
    TOOLTIP_KWARGS = {
        "label": {"label": "Label"},
        "coords": True,
        "index": True,
    }

    def __init__(self, df_dict, **kwargs):
        """
        ???+ note "Constructor shared by all child classes."
            | Param       | Type   | Description                  |
            | :---------- | :----- | :--------------------------- |
            | `df_dict`   | `dict` | `str` -> `DataFrame` mapping |
            | `**kwargs`  |        | forwarded to `bokeh.plotting.figure` |

            1. settle the figure settings by using child class defaults & kwargs overrides
            2. settle the glyph settings by using child class defaults
            3. set up dataframes to sync with
            4. create widgets that child classes can override
            5. create data sources the correspond to class-specific data subsets.
            6. initialize a figure under the settings above
        """
        self.figure_kwargs = {
            "tools": STANDARD_PLOT_TOOLS,
            "tooltips": self._build_tooltip(kwargs.pop("tooltips", "")),
            # bokeh recommends webgl for scalability
            "output_backend": "webgl",
        }
        self.figure_kwargs.update(kwargs)
        self.figure = figure(**self.figure_kwargs)
        self.glyph_kwargs = {
            _key: _dict["constant"].copy()
            for _key, _dict in self.__class__.SUBSET_GLYPH_KWARGS.items()
        }
        self._setup_dfs(df_dict)
        self._setup_sources()
        self._setup_widgets()
        self._setup_status_flags()

    @classmethod
    def from_dataset(cls, dataset, subset_mapping, *args, **kwargs):
        """
        ???+ note "Alternative constructor from a `SupervisableDataset`."
            | Param            | Type   | Description                  |
            | :--------------- | :----- | :--------------------------- |
            | `dataset`        | `SupervisableDataset` | dataset with `DataFrame`s |
            | `subset_mapping` | `dict` | `dataset` -> `explorer` subset mapping |
            | `*args`          |        | forwarded to the constructor |
            | `**kwargs`       |        | forwarded to the constructor |
        """
        # local import to avoid import cycles
        from hover.core.dataset import SupervisableDataset

        assert isinstance(dataset, SupervisableDataset)
        df_dict = {_v: dataset.dfs[_k] for _k, _v in subset_mapping.items()}
        explorer = cls(df_dict, *args, **kwargs)
        explorer.link_dataset(dataset)
        return explorer

    def link_dataset(self, dataset):
        """
        ???+ note "Get tied to a dataset, which is common for explorers."
        """
        if not hasattr(self, "linked_dataset"):
            self.linked_dataset = dataset
        else:
            assert self.linked_dataset is dataset, "cannot link to two datasets"

    def view(self):
        """
        ???+ note "Define the high-level visual layout of the whole explorer."
        """
        from bokeh.layouts import column

        return column(self._layout_widgets(), self.figure)

    def _build_tooltip(self, specified):
        """
        ???+ note "Define a windowed tooltip which shows inspection details."
            | Param            | Type   | Description                  |
            | :--------------- | :----- | :--------------------------- |
            | `specified`      | `str`  | user-specified HTML          |

            Note that this is a method rather than a class attribute because
            child classes may involve instance attributes in the tooltip.
        """
        if not specified:
            return bokeh_hover_tooltip(**self.__class__.TOOLTIP_KWARGS)
        return specified

    def _setup_widgets(self):
        """
        ???+ note "High-level function creating widgets for interactive functionality."
        """
        self._info("Setting up widgets")
        self._dynamic_widgets = OrderedDict()
        self._dynamic_callbacks = OrderedDict()
        self._dynamic_resources = OrderedDict()
        self._setup_search_widgets()
        self._setup_selection_option()
        self._setup_subset_toggle()
        self._setup_axes_dropdown()

    @abstractmethod
    def _layout_widgets(self):
        """
        ???+ note "Define the low-level layout of widgets."

        """
        pass

    @abstractmethod
    def _setup_search_widgets(self):
        """
        ???+ note "Define how to search data points."
            Left to child classes that have a specific feature format.
        """
        pass

    def _setup_status_flags(self):
        """
        ???+ note "Status flags to permit or forbid certain operations."
        """
        self.status_flags = {
            "selecting": False,
            "selection_syncing_out": False,
        }

        def update_selecting_status(event):
            self.status_flags["selecting"] = not event.final

        self.figure.on_event(SelectionGeometry, update_selecting_status)

    def _setup_selection_option(self):
        """
        ???+ note "Create a group of checkbox(es) for advanced selection options."
        """
        from bokeh.models import RadioGroup

        self.selection_option_box = RadioGroup(
            labels=["keep selecting: none", "union", "intersection", "difference"],
            active=0,
        )

    def _setup_subset_toggle(self):
        """
        ???+ note "Create a group of buttons for toggling which data subsets to show."
        """
        from bokeh.models import CheckboxButtonGroup, Div
        from bokeh.layouts import column

        data_keys = list(self.__class__.SUBSET_GLYPH_KWARGS.keys())
        self.data_key_button_group = CheckboxButtonGroup(
            labels=data_keys, active=list(range(len(data_keys)))
        )
        self.data_key_button_group_help = Div(text="Toggle data subset display")
        self.subset_toggle_widget_column = column(
            self.data_key_button_group_help, self.data_key_button_group
        )

        def update_data_key_display():
            subsets = self.data_key_button_group.active
            visible_keys = {self.data_key_button_group.labels[idx] for idx in subsets}
            for _renderer in self.figure.renderers:
                # if the renderer has a name "on the list", update its visibility
                if _renderer.name in self.__class__.SUBSET_GLYPH_KWARGS.keys():
                    _renderer.visible = _renderer.name in visible_keys

        # store the callback (useful, for example, during automated tests) and link it
        self._callback_subset_display = update_data_key_display
        self.data_key_button_group.on_change(
            "active", lambda attr, old, new: update_data_key_display()
        )

    def _setup_axes_dropdown(self):
        """
        ???+ note "Find embedding fields and allow any of them to be set as the x or y axis."
        """
        from bokeh.models import Dropdown

        embed_cols = self.find_embedding_fields()
        init_x, init_y = embed_cols[:2]
        self.dropdown_x_axis = Dropdown(label=f"X coord: {init_x}", menu=embed_cols)
        self.dropdown_y_axis = Dropdown(label=f"Y coord: {init_y}", menu=embed_cols)

        def change_x(event):
            self.dropdown_x_axis.label = f"X coord: {event.item}"
            for _renderer in self.figure.renderers:
                _renderer.glyph.x = event.item

        def change_y(event):
            self.dropdown_y_axis.label = f"Y coord: {event.item}"
            for _renderer in self.figure.renderers:
                _renderer.glyph.y = event.item

        self.dropdown_x_axis.on_click(change_x)
        self.dropdown_y_axis.on_click(change_y)

        # consider allowing dynamic menu refreshment
        # def refresh_axes_list():
        #    embed_cols = self.find_embedding_fields()
        #    self.dropdown_x_axis.menu = embed_cols[:]
        #    self.dropdown_y_axis.menu = embed_cols[:]

    def value_patch_by_slider(self, col_original, col_patch, **kwargs):
        """
        ???+ note "Allow source values to be dynamically patched through a slider."
            | Param            | Type   | Description                  |
            | :--------------- | :----- | :--------------------------- |
            | `col_original`   | `str`  | column of values before the patch |
            | `col_patch`      | `str`  | column of list of values to use as patches |
            | `**kwargs`       |        | forwarded to the slider |

            [Reference](https://github.com/bokeh/bokeh/blob/2.4.2/examples/howto/patch_app.py)
        """
        # add a patch slider to widgets, if none exist
        if "patch_slider" not in self._dynamic_widgets:
            slider = Slider(start=0, end=1, value=0, step=1, **kwargs)
            slider.disabled = True
            self._dynamic_widgets["patch_slider"] = slider
        else:
            slider = self._dynamic_widgets["patch_slider"]

        # create a slider-adjusting callback exposed to the outside
        def adjust_slider():
            """
            Infer slider length from the number of patch values.
            """
            num_patches = None
            for _key, _df in self.dfs.items():
                assert (
                    col_patch in _df.columns
                ), f"Subset {_key} expecting column {col_patch} among columns, got {_df.columns}"
                # find all array lengths; note that the data subset can be empty
                _num_patches_seen = _df[col_patch].apply(len).values
                assert (
                    len(set(_num_patches_seen)) <= 1
                ), f"Expecting consistent number of patches, got {_num_patches_seen}"
                _num_patches = _num_patches_seen[0] if _df.shape[0] > 0 else None

                # if a previous subset has implied the number of patches, run a consistency check
                if num_patches is None:
                    num_patches = _num_patches
                else:
                    assert (
                        num_patches == _num_patches
                    ), f"Conflicting number of patches: {num_patches} vs {_num_patches}"

            assert num_patches >= 2, f"Expecting at least 2 patches, got {num_patches}"
            slider.end = num_patches - 1
            slider.disabled = False

        self._dynamic_callbacks["adjust_patch_slider"] = adjust_slider

        # create the callback for patching values
        def update_patch(attr, old, new):
            for _key, _df in self.dfs.items():
                # calculate the patch corresponding to slider value
                _value = [_arr[new] for _arr in _df[col_patch].values]
                _slice = slice(_df.shape[0])
                _patch = {col_original: [(_slice, _value)]}
                self.sources[_key].patch(_patch)

        slider.on_change("value", update_patch)
        self._good(f"Patching {col_original} using {col_patch}")

    def _mandatory_column_defaults(self):
        """
        ???+ note "Mandatory columns and default values."

            If default value is None, will raise exception if the column is not found.
        """
        return {_col: None for _col in self.__class__.MANDATORY_COLUMNS}

    def _setup_dfs(self, df_dict, copy=False):
        """
        ???+ note "Check and store DataFrames **by reference by default**."
            Intended to be extended in child classes for pre/post processing.

            | Param       | Type   | Description                  |
            | :---------- | :----- | :--------------------------- |
            | `df_dict`   | `dict` | `str` -> `DataFrame` mapping |
            | `copy`      | `bool` | whether to copy `DataFrame`s |
        """
        self._info("Setting up DataFrames")
        supplied_keys = set(df_dict.keys())
        expected_keys = set(self.__class__.SUBSET_GLYPH_KWARGS.keys())

        # perform high-level df key checks
        expected_and_supplied = supplied_keys.intersection(expected_keys)
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

        # assign df with column checks
        self.dfs = dict()
        mandatory_col_to_default = self._mandatory_column_defaults()
        for _key in expected_and_supplied:
            _df = df_dict[_key]
            for _col, _default in mandatory_col_to_default.items():
                # column exists: all good
                if _col in _df.columns:
                    continue
                # no default value: column must be explicitly provided
                if _default is None:
                    # edge case: DataFrame has zero rows
                    _msg = f"Expecting column '{_col}' from {_key} df: found {list(_df.columns)}"
                    assert _df.shape[0] == 0, _msg
                # default value available, will use it to create column
                else:
                    _df[_col] = _default
            self.dfs[_key] = _df.copy() if copy else _df

        # expected dfs must be present
        for _key in expected_not_supplied:
            _df = pd.DataFrame(columns=list(mandatory_col_to_default.keys()))
            self.dfs[_key] = _df

    def _setup_sources(self):
        """
        ???+ note "Create, **(not update)**, `ColumnDataSource` objects."
            Intended to be extended in child classes for pre/post processing.
        """
        self._info("Setting up sources")
        self.sources = {_key: ColumnDataSource(_df) for _key, _df in self.dfs.items()}
        self._postprocess_sources()

        # initialize attributes that couple with sources
        # extra columns for dynamic plotting
        self._extra_source_cols = defaultdict(dict)

        self._setup_selection_tools()

    def _setup_subroutine_selection_callback_queue(self):
        """
        ???+ note "For dynamically assigned callbacks triggered by making a selection on the figure."
        """
        all_stages = self.__class__.SELECTION_PROCESSING_STAGES
        stage_to_order = {_stage: _i for _i, _stage in enumerate(all_stages)}

        self._selection_callbacks = {_k: RootUnionFind(set()) for _k in all_stages}

        def stages_callback(*stages):
            prev_order = -1
            for _stage in stages:
                _order = stage_to_order[_stage]
                assert _order > prev_order, f"Misordered stage sequence {stages}"
                for _callback in self._selection_callbacks[_stage].data:
                    _callback()

        self._selection_stages_callback = stages_callback

        self.figure.on_event(
            SelectionGeometry,
            lambda event: stages_callback(*all_stages) if event.final else None,
        )

        def register_selection_callback(stage, callback):
            assert (
                stage in all_stages
            ), f"Invalid stage: {stage}, expected one of {all_stages}"
            self._selection_callbacks[stage].data.add(callback)

        self._register_selection_callback = register_selection_callback

    def _setup_subroutine_selection_store(self):
        """
        ???+ note "Subroutine of `_setup_selection_tools`."
            Setup callbacks that interact with manual selections.
        """

        def store_selection():
            """
            Keep track of the last manual selection.
            Useful for applying cumulation / filters dynamically.
            """
            # determine selection mode
            selection_option_code = self.selection_option_box.active

            for _key, _source in self.sources.items():
                _selected = _source.selected.indices
                # use sets' in-place methods instead of assignment
                if selection_option_code == 1:
                    self._last_selections[_key].data.update(_selected)
                elif selection_option_code == 2:
                    self._last_selections[_key].data.intersection_update(_selected)
                elif selection_option_code == 3:
                    self._last_selections[_key].data.difference_update(_selected)
                else:
                    assert selection_option_code == 0
                    self._last_selections[_key].data.clear()
                    self._last_selections[_key].data.update(_selected)
                _source.selected.indices = list(self._last_selections[_key].data)

        def restore_selection():
            """
            Set current selection to the last manual selection.
            Useful for applying cumulation / filters dynamically.
            """
            for _key, _source in self.sources.items():
                _source.selected.indices = list(self._last_selections[_key].data)

        self._store_selection = store_selection
        self._register_selection_callback("save", store_selection)
        self._register_selection_callback("load", restore_selection)

    def _setup_subroutine_selection_filter(self):
        """
        ???+ note "Subroutine of `_setup_selection_tools`."
            Setup callbacks that interact with selection filters.
        """

        def trigger_selection_filters(subsets=None):
            """
            Filter selection indices on specified subsets.
            """
            if subsets is None:
                subsets = self.sources.keys()
            else:
                assert set(subsets).issubset(
                    self.sources.keys()
                ), f"Expected subsets from {self.sources.keys()}"

            for _key in subsets:
                _selected = set(self.sources[_key].selected.indices[:])
                for _func in self._selection_filters[_key].data:
                    _selected = _func(_selected, _key)
                self.sources[_key].selected.indices = sorted(_selected)

        self._register_selection_callback("write", trigger_selection_filters)

    def _setup_subroutine_selection_reset(self):
        """
        ???+ note "Subroutine of `_setup_selection_tools`."
            Setup callbacks for scenarios where the selection should be reset.
            For example, when the plot sources have changed.
        """

        def reset_selection():
            """
            Clear last manual selections and source selections.
            Useful during post-processing of refreshed data source.
            Not directly defined as a method because of `self._last_selections`.
            """
            for _key, _source in self.sources.items():
                self._last_selections[_key].data.clear()
                _source.selected.indices = []

        self._reset_selection = reset_selection

    def _setup_selection_tools(self):
        """
        ???+ note "Create data structures and callbacks for dynamic selections."
            Useful for linking and filtering selections across explorers.
        """
        # store the last manual selections
        self._last_selections = {
            _key: RootUnionFind(set()) for _key in self.sources.keys()
        }
        # store commutative, idempotent index filters
        self._selection_filters = {
            _key: RootUnionFind(set()) for _key in self.sources.keys()
        }

        self._setup_subroutine_selection_callback_queue()
        self._setup_subroutine_selection_store()
        self._setup_subroutine_selection_filter()
        self._setup_subroutine_selection_reset()

    def _update_sources(self):
        """
        ???+ note "Update the sources with the corresponding dfs."
            Note that the shapes and fields of sources are overriden.
            Thus supplementary fields (those that do not exist in the dfs),
            such as dynamic plotting kwargs, need to be re-assigned.
        """
        for _key in self.dfs.keys():
            self.sources[_key].data = self.dfs[_key]
        self._postprocess_sources()

        # reset selections now that source indices may have changed
        self._reset_selection()

        # reset attribute values that couple with sources
        for _key in self.sources.keys():
            _num_points = len(self.sources[_key].data["label"])
            # add extra columns
            for _col, _fill_value in self._extra_source_cols[_key].items():
                self.sources[_key].add([_fill_value] * _num_points, _col)

            # clear last selection but keep the set object
            self._last_selections[_key].data.clear()
            # DON'T DO: self._last_selections = {_key: set() for _key in self.sources.keys()}

    def _postprocess_sources(self):
        """
        ???+ note "Infer source attributes from the dfs, without altering the dfs."
            Useful for assigning dynamic glyph attributes, similarly to `activate_search()`.
        """
        pass

    def activate_search(self):
        """
        ???+ note "Assign search response callbacks to search results. Child methods should bind those callbacks to search widgets."

            This is a parent method which takes care of common denominators of parent methods.

            Child methods may inherit the logic here and preprocess/postprocess as needed.
        """
        self._subroutine_search_create_callbacks()
        self._subroutine_search_activate_callbacks()

    def _subroutine_search_create_callbacks(self):
        """
        ???+ note "Create search callback functions based on feature attributes."
        """
        # allow dynamically updated search response through dict element retrieval
        self._dynamic_callbacks["search_response"] = dict()

        def search_base_response(attr, old, new):
            for _subset in self.sources.keys():
                _func = self._dynamic_callbacks["search_response"].get(_subset, blank)
                _func(attr, old, new)
            return

        self.search_base_response = search_base_response

        for _key, _dict in self.__class__.SUBSET_GLYPH_KWARGS.items():
            # create a field that holds search results that could be used elsewhere
            _num_points = len(self.sources[_key].data[self.__class__.PRIMARY_FEATURE])
            self._extra_source_cols[_key][SEARCH_SCORE_FIELD] = 0
            self.sources[_key].add([0] * _num_points, SEARCH_SCORE_FIELD)

            # make attributes respond to search
            for _, _params in _dict["search"].items():
                _updated_kwargs = self._subroutine_search_source_change(
                    _key,
                    self.glyph_kwargs[_key],
                    altered_param=_params,
                )
                self.glyph_kwargs[_key].clear()
                self.glyph_kwargs[_key].update(_updated_kwargs)

    def _subroutine_search_activate_callbacks(self):
        """
        ???+ note "Activate search callback functions by binding them to widgets."
        """
        for _widget in self._search_watch_widgets():
            _widget.on_change("value", self.search_base_response)
            self._info(f"activated search base response on {_widget}")

    @abstractmethod
    def _search_watch_widgets(self):
        """
        ???+ note "Widgets to trigger search callbacks automatically, which can be different across subclasses."

            Intended for binding callback functions to widgets.
        """

    @abstractmethod
    def _validate_search_input(self):
        """
        ???+ note "Check the search input, skipping callbacks if it's invalid."
        """
        pass

    @abstractmethod
    def _get_search_score_function(self):
        """
        ???+ note "Dynamically create a single-argument scoring function."
        """
        pass

    def _subroutine_search_source_change(
        self, subset, kwargs, altered_param=("size", 10, 5, 7)
    ):
        """
        ???+ note "Subroutine of `activate_search()` on a specific subset."
            Modifies the plotting source in-place.

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `subset`        | `str`   | the subset to activate search on |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
        """
        assert isinstance(kwargs, dict)
        updated_kwargs = kwargs.copy()

        feature_key = self.__class__.PRIMARY_FEATURE
        param_key, param_pos, param_neg, param_default = altered_param
        initial_num = len(self.sources[subset].data[feature_key])
        self.sources[subset].add([param_default] * initial_num, param_key)
        self._extra_source_cols[subset][param_key] = param_default

        updated_kwargs[param_key] = param_key

        def score_to_param(score):
            if score > 0:
                return param_pos
            elif score == 0:
                return param_default
            else:
                return param_neg

        def search_response(attr, old, new):
            valid_flag = self._validate_search_input()
            if not valid_flag:
                return
            score_func = self._get_search_score_function()

            patch_slice = slice(len(self.sources[subset].data[feature_key]))
            features = self.sources[subset].data[self.__class__.PRIMARY_FEATURE]
            # score_func is potentially slow; track its progress
            # search_scores = list(map(score_func, tqdm(features, desc="Search score")))
            search_scores = list(map(score_func, features))
            search_params = list(map(score_to_param, search_scores))
            self.sources[subset].patch(
                {SEARCH_SCORE_FIELD: [(patch_slice, search_scores)]}
            )
            self.sources[subset].patch({param_key: [(patch_slice, search_params)]})
            return

        # assign dynamic callback
        self._dynamic_callbacks["search_response"][subset] = search_response
        return updated_kwargs

    def _prelink_check(self, other):
        """
        ???+ note "Sanity check before linking two explorers."
            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
        """
        assert other is not self, "Self-loops are fordidden"
        assert isinstance(other, BokehBaseExplorer), "Must link to BokehBaseExplorer"

    def link_selection(self, other, subset_mapping):
        """
        ???+ note "Synchronize the selection mechanism between sources."

            This includes:
            -   the selected indices between subsets
            -   callbacks associated with selections
            -   selection option values in the widgets

            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
            | `subset_mapping` | `dict` | mapping of subsets from `self` to `other` |
        """
        self._prelink_check(other)
        self._subroutine_link_selection_callbacks(other)
        self._subroutine_link_selection_indices(other, subset_mapping)
        self._subroutine_link_selection_options(other)

    def _subroutine_link_selection_callbacks(self, other):
        """
        ???+ note "Subroutine of `link_selection`."

            Union the callbacks triggered by selection event.

            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
        """
        # link selection callbacks (pointing to the same set)
        for _k in self.__class__.SELECTION_PROCESSING_STAGES:
            self._selection_callbacks[_k].data.update(
                other._selection_callbacks[_k].data
            )
            self._selection_callbacks[_k].union(other._selection_callbacks[_k])

    def _subroutine_link_selection_indices(self, other, subset_mapping):
        """
        ???+ note "Subroutine of `link_selection`."

            Synchronize the manually selected indices and actually selected ones.

            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
            | `subset_mapping` | `dict` | mapping of subsets from `self` to `other` |
        """

        def link_selected_indices(kl, kr):
            sl, sr = self.sources[_kl], other.sources[_kr]

            # acyclic, DFS-like syncs
            def left_to_right(attr, old, new):
                # "acyclic"
                if other.status_flags["selection_syncing_out"]:
                    return

                # "DFS-like"
                if set(sr.selected.indices) ^ set(sl.selected.indices):
                    self.status_flags["selection_syncing_out"] = True
                    sr.selected.indices = sl.selected.indices[:]
                    self.status_flags["selection_syncing_out"] = False

            def right_to_left(attr, old, new):
                # "acyclic"
                if self.status_flags["selection_syncing_out"]:
                    return

                # "DFS-like"
                if set(sl.selected.indices) ^ set(sr.selected.indices):
                    other.status_flags["selection_syncing_out"] = True
                    sl.selected.indices = sr.selected.indices[:]
                    other.status_flags["selection_syncing_out"] = False

            sl.selected.on_change("indices", left_to_right)
            sr.selected.on_change("indices", right_to_left)

        for _kl, _kr in subset_mapping.items():
            # link last manual selections (pointing to the same set)
            self._last_selections[_kl].union(other._last_selections[_kr])

            # link selected indices; these are used by bokeh, not UnionFind-able
            link_selected_indices(self.sources[_kl], other.sources[_kr])

    def _subroutine_link_selection_options(self, other):
        """
        ???+ note "Subroutine of `link_selection`."

            Synchronize the option widget values associated with selection.

            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
        """
        # link selection option values
        def option_lr(attr, old, new):
            other.selection_option_box.active = self.selection_option_box.active

        def option_rl(attr, old, new):
            self.selection_option_box.active = other.selection_option_box.active

        self.selection_option_box.on_change("active", option_lr)
        other.selection_option_box.on_change("active", option_rl)

    def link_xy_range(self, other):
        """
        ???+ note "Synchronize plotting ranges on the xy-plane."
            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
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
        ???+ note "Plot something onto the figure."
            Implemented in child classes based on their functionalities.
            | Param      | Type  | Description           |
            | :--------- | :---- | :-------------------- |
            | `*args`    |       | left to child classes |
            | `**kwargs` |       | left to child classes |
        """
        pass

    def find_embedding_fields(self):
        """
        ???+ note "Find embedding fields from dataframes."

            Intended for scenarios where the embedding is higher than two-dimensional.
        """
        embedding_cols = None
        for _key, _df in self.dfs.items():
            # edge case: dataframe is empty
            if _df.shape[0] == 0:
                continue
            # automatically find embedding columns
            _emb_cols = sorted(filter(is_embedding_field, _df.columns))
            if embedding_cols is None:
                embedding_cols = _emb_cols
            else:
                # embedding columns must be the same across subsets
                assert embedding_cols == _emb_cols, "Inconsistent embedding columns"
        assert (
            len(embedding_cols) >= 2
        ), f"Expected at least two embedding columns, found {embedding_cols}"
        return embedding_cols

    def auto_color_mapping(self):
        """
        ???+ note "Find all labels and an appropriate color for each."
        """
        from hover.utils.bokeh_helper import auto_label_color

        labels = set()
        for _key in self.dfs.keys():
            labels = labels.union(set(self.dfs[_key]["label"].values))

        return auto_label_color(labels)

    # def auto_legend_correction(self):
    #    """
    #    ???+ note "Find legend items and deduplicate by label, keeping the last glyph / legend item of each label."
    #        This is to resolve duplicate legend items due to automatic legend_group and incremental plotting.
    #    """
    #    from collections import OrderedDict
    #
    #    if not hasattr(self.figure, "legend"):
    #        self._fail("Attempting auto_legend_correction when there is no legend")
    #        return
    #    # extract all items and start over
    #    items = self.figure.legend.items[:]
    #    self.figure.legend.items.clear()
    #
    #    # use one item to hold all renderers matching its label
    #    label_to_item = OrderedDict()
    #
    #    # deduplication
    #    for _item in items:
    #        _label = _item.label.get("value", "")
    #        label_to_item[_label] = _item
    #
    #        # WARNING: the current implementation discards renderer references.
    #        # This could be for the best because renderers add up their glyphs to the legend item.
    #        # To keep renderer references, see this example:
    #        # if _label not in label_to_item.keys():
    #        #    label_to_item[_label] = _item
    #        # else:
    #        #    label_to_item[_label].renderers.extend(_item.renderers)
    #
    #    self.figure.legend.items = list(label_to_item.values())
    #
    #    return
    #
    # @staticmethod
    # def auto_legend(method):
    #    """
    #    ???+ note "Decorator that handles legend pre/post-processing issues."
    #        Usage:
    #
    #        ```python
    #        # in a child class
    #
    #        @BokehBaseExplorer.auto_legend
    #        def plot(self, *args, **kwargs):
    #            # put code here
    #            pass
    #        ```
    #    """
    #    from functools import wraps
    #
    #    @wraps(method)
    #    def wrapped(ref, *args, **kwargs):
    #        if hasattr(ref.figure, "legend"):
    #            if hasattr(ref.figure.legend, "items"):
    #                ref.figure.legend.items.clear()
    #
    #        retval = method(ref, *args, **kwargs)
    #
    #        ref.auto_legend_correction()
    #
    #        return retval
    #
    #    return wrapped
