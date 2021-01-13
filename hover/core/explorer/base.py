"""
???+ info "Docstring"
    Base class(es) for ALL explorer implementations.
"""
from abc import ABC, abstractmethod
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10, Category20
from hover import module_config
from hover.core import Loggable
from .local_config import bokeh_hover_tooltip

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


class BokehBaseExplorer(Loggable, ABC):
    """
    ???+ info "Docstring"
        Base class for exploring data.

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
        ???+ info "Docstring"
            Operations shared by all child classes.

            - settle the figure settings by using child class defaults & kwargs overrides
            - settle the glyph settings by using child class defaults
            - create widgets that child classes can override
            - create data sources the correspond to class-specific data subsets.
            - activate builtin search callbacks depending on the child class.
            - create a (typically) blank figure under such settings
        """
        self.figure_kwargs = {
            "tools": STANDARD_PLOT_TOOLS,
            "tooltips": self._build_tooltip(),
            # bokeh recommends webgl for scalability
            "output_backend": "webgl",
        }
        self.figure_kwargs.update(kwargs)
        self.glyph_kwargs = {
            _key: _dict["constant"].copy()
            for _key, _dict in self.__class__.SUBSET_GLYPH_KWARGS.items()
        }
        self._setup_widgets()
        self._setup_dfs(df_dict)
        self._setup_sources()
        self._activate_search_builtin()
        self.figure = figure(**self.figure_kwargs)

    @classmethod
    def from_dataset(cls, dataset, subset_mapping, *args, **kwargs):
        """
        ???+ info "Docstring"
            Construct from a SupervisableDataset.
        """
        # local import to avoid import cycles
        from hover.core.dataset import SupervisableDataset

        assert isinstance(dataset, SupervisableDataset)
        df_dict = {_v: dataset.dfs[_k] for _k, _v in subset_mapping.items()}
        return cls(df_dict, *args, **kwargs)

    def view(self):
        """
        ???+ info "Docstring"
            Define the layout of the whole explorer.
        """
        from bokeh.layouts import column

        return column(self._layout_widgets(), self.figure)

    def _build_tooltip(self):
        """
        ???+ info "Docstring"
            Define a windowed tooltip which shows inspection details.

            Note that this is a method rather than a class attribute because
            child classes may involve instance attributes in the tooltip.
        """
        return bokeh_hover_tooltip(**self.__class__.TOOLTIP_KWARGS)

    def _setup_widgets(self):
        """
        ???+ info "Docstring"
            Prepare widgets for interactive functionality.

            Create positive/negative text search boxes.
        """
        self._info("Setting up widgets")
        self._setup_search_highlight()
        self._setup_subset_toggle()

    @abstractmethod
    def _layout_widgets(self):
        """
        ???+ info "Docstring"
            Define the layout of widgets.
        """
        pass

    @abstractmethod
    def _setup_search_highlight(self):
        """
        ???+ info "Docstring"
            Left to child classes that have a specific data format.
        """
        pass

    def _setup_subset_toggle(self):
        """
        ???+ info "Docstring"
            Widgets for toggling which data subsets to show.
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

    def _setup_dfs(self, df_dict, copy=False):
        """
        ???+ info "Docstring"
            Check and store DataFrames BY REFERENCE BY DEFAULT.

            Intended to be extended in child classes for pre/post processing.
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
        ???+ info "Docstring"
            Create (NOT UPDATE) ColumnDataSource objects.

            Intended to be extended in child classes for pre/post processing.
        """
        self._info("Setting up sources")
        self.sources = {_key: ColumnDataSource(_df) for _key, _df in self.dfs.items()}

    def _update_sources(self):
        """
        ???+ info "Docstring"
            Update the sources with the corresponding dfs.

            Note that it seems mandatory to re-activate the search widgets.
            This is because the source loses plotting kwargs.
        """
        for _key in self.dfs.keys():
            self.sources[_key].data = self.dfs[_key]
        self._activate_search_builtin(verbose=False)

    def _activate_search_builtin(self, verbose=True):
        """
        ???+ info "Docstring"
            Assign Highlighting callbacks to search results in a manner built into the class.
            Typically called once during initialization.

            Note that this is a template method which heavily depends on class attributes.
        """
        for _key, _dict in self.__class__.SUBSET_GLYPH_KWARGS.items():
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

    @abstractmethod
    def activate_search(self, source, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ info "Docstring"
            Left to child classes that have a specific data format.
        """
        pass

    def _prelink_check(self, other):
        """
        ???+ info "Docstring"
            Sanity check before linking two explorers.
        """
        assert other is not self, "Self-loops are fordidden"
        assert isinstance(other, BokehBaseExplorer), "Must link to BokehBaseExplorer"

    def link_selection(self, key, other, other_key):
        """
        ???+ info "Docstring"
            Sync the selected indices between specified sources.
        """
        self._prelink_check(other)
        # link selection in a bidirectional manner
        sl, sr = self.sources[key], other.sources[other_key]
        sl.selected.js_link("indices", sr.selected, "indices")
        sr.selected.js_link("indices", sl.selected, "indices")

    def link_xy_range(self, other):
        """
        ???+ info "Docstring"
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
        ???+ info "Docstring"
            Plot something onto the figure.
        """
        pass

    def auto_labels_palette(self):
        """
        ???+ info "Docstring"
            Find all labels and an appropriate color map.
        """
        labels = set()
        for _key in self.dfs.keys():
            labels = labels.union(set(self.dfs[_key]["label"].values))
        labels.discard(module_config.ABSTAIN_DECODED)
        labels = sorted(labels, reverse=True)

        assert len(labels) <= 20, "Too many labels to support (max at 20)"
        palette = Category10[10] if len(labels) <= 10 else Category20[20]
        return labels, palette

    def auto_legend_correction(self):
        """
        ???+ info "Docstring"
            Find legend items and deduplicate by label, keeping the last glyph / legend item of each label.

            This is to resolve duplicate legend items due to automatic legend_group and incremental plotting.
        """
        from collections import OrderedDict

        if not hasattr(self.figure, "legend"):
            self._fail("Attempting auto_legend_correction when there is no legend")
            return
        # extract all items and start over
        items = self.figure.legend.items[:]
        self.figure.legend.items.clear()

        # use one item to hold all renderers matching its label
        label_to_item = OrderedDict()

        # deduplication
        for _item in items:
            _label = _item.label.get("value", "")
            label_to_item[_label] = _item

            # WARNING: the current implementation discards renderer references.
            # This could be for the best because renderers add up their glyphs to the legend item.
            # To keep renderer references, see this example:
            # if _label not in label_to_item.keys():
            #    label_to_item[_label] = _item
            # else:
            #    label_to_item[_label].renderers.extend(_item.renderers)

        self.figure.legend.items = list(label_to_item.values())

        return

    @staticmethod
    def auto_legend(method):
        """
        ???+ info "Docstring"
            Decorator that enforces the legend after each call of the decorated method.
        """
        from functools import wraps

        @wraps(method)
        def wrapped(ref, *args, **kwargs):
            if hasattr(ref.figure, "legend"):
                if hasattr(ref.figure.legend, "items"):
                    ref.figure.legend.items.clear()

            retval = method(ref, *args, **kwargs)

            ref.auto_legend_correction()
            return retval

        return wrapped
