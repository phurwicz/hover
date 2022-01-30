"""
???+ note "Base class(es) for ALL explorer implementations."
"""
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from bokeh.events import SelectionGeometry
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from hover.core import Loggable
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
    # navigate changes
    "undo",
    "redo",
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

    MANDATORY_COLUMNS = ["label", "x", "y"]
    TOOLTIP_KWARGS = {"label": True, "coords": True, "index": True}

    def __init__(self, df_dict, **kwargs):
        """
        ???+ note "Constructor shared by all child classes."
            | Param       | Type   | Description                  |
            | :---------- | :----- | :--------------------------- |
            | `df_dict`   | `dict` | `str` -> `DataFrame` mapping |
            | `**kwargs`  |        | forwarded to `bokeh.plotting.figure` |

            1. settle the figure settings by using child class defaults & kwargs overrides
            2. settle the glyph settings by using child class defaults
            3. create widgets that child classes can override
            4. create data sources the correspond to class-specific data subsets.
            5. activate builtin search callbacks depending on the child class.
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
        self._activate_search_builtin()

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
        return cls(df_dict, *args, **kwargs)

    def view(self):
        """
        ???+ note "Define the high-level visual layout of the whole explorer."
        """
        from bokeh.layouts import column

        return column(self._layout_widgets(), self.figure)

    def _build_tooltip(self, extra):
        """
        ???+ note "Define a windowed tooltip which shows inspection details."
            | Param            | Type   | Description                  |
            | :--------------- | :----- | :--------------------------- |
            | `extra`          | `str`  | user-supplied extra HTML |

            Note that this is a method rather than a class attribute because
            child classes may involve instance attributes in the tooltip.
        """
        standard = bokeh_hover_tooltip(**self.__class__.TOOLTIP_KWARGS)
        return f"{standard}\n{extra}"

    def _setup_widgets(self):
        """
        ???+ note "High-level function creating widgets for interactive functionality."
        """
        self._info("Setting up widgets")
        self._dynamic_widgets = OrderedDict()
        self._dynamic_callbacks = OrderedDict()
        self._setup_search_highlight()
        self._setup_selection_option()
        self._setup_subset_toggle()

    @abstractmethod
    def _layout_widgets(self):
        """
        ???+ note "Define the low-level layout of widgets."

        """
        pass

    @abstractmethod
    def _setup_search_highlight(self):
        """
        ???+ note "Define how to search and highlight data points."
            Left to child classes that have a specific feature format.
        """
        pass

    def _setup_selection_option(self):
        """
        ???+ note "Create a group of checkbox(es) for advanced selection options."
        """
        from bokeh.models import CheckboxGroup

        self.selection_option_box = CheckboxGroup(
            labels=["cumulative selection"], active=[]
        )

    def _setup_subset_toggle(self):
        """
        ???+ note "Create a group of buttons for toggling which data subsets to show."
        """
        from bokeh.models import CheckboxButtonGroup

        data_keys = list(self.__class__.SUBSET_GLYPH_KWARGS.keys())
        self.data_key_button_group = CheckboxButtonGroup(
            labels=data_keys, active=list(range(len(data_keys)))
        )

        def update_data_key_display(active):
            visible_keys = {self.data_key_button_group.labels[idx] for idx in active}
            for _renderer in self.figure.renderers:
                # if the renderer has a name "on the list", update its visibility
                if _renderer.name in self.__class__.SUBSET_GLYPH_KWARGS.keys():
                    _renderer.visible = _renderer.name in visible_keys

        # store the callback (useful, for example, during automated tests) and link it
        self._callback_subset_display = lambda: update_data_key_display(
            self.data_key_button_group.active
        )
        self.data_key_button_group.on_click(update_data_key_display)

    def value_patch(self, col_original, col_patch, **kwargs):
        """
        ???+ note "Allow source values to be dynamically patched through a slider."
            | Param            | Type   | Description                  |
            | :--------------- | :----- | :--------------------------- |
            | `col_original`   | `str`  | column of values before the patch |
            | `col_patch`      | `str`  | column of list of values to use as patches |
            | `**kwargs`       |        | forwarded to the slider |

            [Reference](https://github.com/bokeh/bokeh/blob/2.3.0/examples/howto/patch_app.py)
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
                    if _col not in _df.columns:
                        # edge case: DataFrame has zero rows
                        assert (
                            _df.shape[0] == 0
                        ), f"Missing column '{_col}' from non-empty {_key} DataFrame: found {list(_df.columns)}"
                        _df[_col] = None

                self.dfs[_key] = _df.copy() if copy else _df

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

    def _setup_subroutine_selection_store(self):
        """
        ???+ note "Subroutine of `_setup_selection_tools`."
            Setup callbacks that interact with manual selections.
        """

        def cumulative_selection_flag():
            """
            Determine whether cumulative selection is enabled.
            """
            return bool(0 in self.selection_option_box.active)

        def store_selection():
            """
            Keep track of the last manual selection.
            Useful for applying cumulation / filters dynamically.
            """
            # store selection indices
            for _key, _source in self.sources.items():
                _selected = _source.selected.indices
                # use clear() and update() instead of assignment to keep clean references
                if not cumulative_selection_flag():
                    self._last_selections[_key].data.clear()
                    self._last_selections[_key].data.update(_selected)
                else:
                    self._last_selections[_key].data.update(_selected)
                    _source.selected.indices = list(self._last_selections[_key].data)

        self._store_selection = store_selection
        self.figure.on_event(
            SelectionGeometry,
            lambda event: self._store_selection() if event.final else None,
        )

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
                _selected = self._last_selections[_key].data
                for _func in self._selection_filters[_key].data:
                    _selected = _func(_selected, _key)
                self.sources[_key].selected.indices = list(_selected)

        # keep reference to trigger_selection_filter() for further access
        # for example, toggling filters should call the trigger
        self._trigger_selection_filters = trigger_selection_filters
        self.figure.on_event(
            SelectionGeometry,
            lambda event: self._trigger_selection_filters() if event.final else None,
        )

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

    def _activate_search_builtin(self, verbose=True):
        """
        ???+ note "Assign Highlighting callbacks to search results in a manner built into the class."
            Typically called once during initialization.

            Note that this is a template method which heavily depends on class attributes.
            | Param       | Type   | Description                  |
            | :---------- | :----- | :--------------------------- |
            | `verbose`   | `bool` | whether to log verbosely     |
        """
        for _key, _dict in self.__class__.SUBSET_GLYPH_KWARGS.items():
            if _key in self.sources.keys():
                # determine responding attributes
                _responding = list(_dict["search"].keys())

                # create a field that holds search results that could be used elsewhere
                _num_points = len(self.sources[_key].data["label"])
                self._extra_source_cols[_key][SEARCH_SCORE_FIELD] = 0
                self.sources[_key].add([0] * _num_points, SEARCH_SCORE_FIELD)

                # make attributes respond to search
                for _flag, _params in _dict["search"].items():
                    self.glyph_kwargs[_key] = self.activate_search(
                        _key,
                        self.glyph_kwargs[_key],
                        altered_param=_params,
                    )
                if verbose:
                    self._info(
                        f"Activated {_responding} on subset {_key} to respond to the search widgets."
                    )

    @abstractmethod
    def activate_search(self, subset, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ note "Left to child classes that have a specific feature format."

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `subset`        | `str`   | the subset to activate search on |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
        """
        pass

    def _prelink_check(self, other):
        """
        ???+ note "Sanity check before linking two explorers."
            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
        """
        assert other is not self, "Self-loops are fordidden"
        assert isinstance(other, BokehBaseExplorer), "Must link to BokehBaseExplorer"

    def link_selection(self, key, other, other_key):
        """
        ???+ note "Synchronize the selected indices between specified sources."
            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `key`   | `str`   | the key of the subset to link  |
            | `other` | `BokehBaseExplorer` | the other explorer |
            | `other_key` | `str` | the key of the other subset  |
        """
        self._prelink_check(other)
        # link selection in a bidirectional manner
        sl, sr = self.sources[key], other.sources[other_key]

        def left_to_right(attr, old, new):
            sr.selected.indices = sl.selected.indices[:]

        def right_to_left(attr, old, new):
            sl.selected.indices = sr.selected.indices[:]

        sl.selected.on_change("indices", left_to_right)
        sr.selected.on_change("indices", right_to_left)

        # link last manual selections (pointing to the same set)
        self._last_selections[key].union(other._last_selections[other_key])

        # link selection filter functions (pointing to the same set)
        self._selection_filters[key].data.update(
            other._selection_filters[other_key].data
        )
        self._selection_filters[key].union(other._selection_filters[other_key])

    def link_selection_options(self, other):
        """
        ???+ note "Synchronize the selection option values between explorers."
            | Param   | Type    | Description                    |
            | :------ | :------ | :----------------------------- |
            | `other` | `BokehBaseExplorer` | the other explorer |
        """

        def left_to_right(attr, old, new):
            other.selection_option_box.active = self.selection_option_box.active[:]

        def right_to_left(attr, old, new):
            self.selection_option_box.active = other.selection_option_box.active[:]

        self.selection_option_box.on_change("active", left_to_right)
        other.selection_option_box.on_change("active", right_to_left)

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
