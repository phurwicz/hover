"""
???+ note "Intermediate classes based on the main feature."
"""
import re
from bokeh.models import TextInput
from bokeh.layouts import column, row
from .base import BokehBaseExplorer


class BokehForText(BokehBaseExplorer):
    """
    ???+ note "`BokehBaseExplorer` with `text` (`str`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) text data in a `text` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "text"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {"label": True, "text": True, "coords": True, "index": True}

    def _setup_search_highlight(self):
        """
        ???+ note "Create positive/negative text search boxes."
        """
        self.search_pos = TextInput(
            title="Text contains: (python regex):",
            width_policy="fit",
            height_policy="fit",
        )
        self.search_neg = TextInput(
            title="Text does not contain:", width_policy="fit", height_policy="fit"
        )

        # allow dynamically updated search response through dict element retrieval
        self._dynamic_callbacks["search_response"] = dict()

        def search_base_response(attr, old, new):
            for _subset in self.sources.keys():
                self._dynamic_callbacks["search_response"].get(
                    _subset, lambda attr, old, new: None
                )(attr, old, new)
            return

        self.search_pos.on_change("value", search_base_response)
        self.search_neg.on_change("value", search_base_response)

    def _layout_widgets(self):
        """
        ???+ note "Define the layout of widgets."
        """
        return column(
            self.search_pos,
            self.search_neg,
            self.data_key_button_group,
            row(*self._dynamic_widgets.values()),
        )

    def activate_search(self):
        """
        ???+ note "Assign Highlighting callbacks to search results."

            No special setup for text since regex search is simple.
        """
        super().activate_search()

    def _get_search_score_function(self):
        """
        ???+ note "Dynamically create a single-argument scoring function."
        """
        pos_regex, neg_regex = self.search_pos.value, self.search_neg.value

        def regex_score(text):
            score = 0
            if len(pos_regex) > 0:
                score += 1 if re.search(pos_regex, text) else -2
            if len(neg_regex) > 0:
                score += -2 if re.search(neg_regex, text) else 1
            return score

        return regex_score


class BokehForAudio(BokehBaseExplorer):
    """
    ???+ note "`BokehBaseExplorer` with `audio` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) audio urls in an `audio` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "audio"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {"label": True, "audio": True, "coords": True, "index": True}

    def _setup_search_highlight(self):
        """
        ???+ help "Help wanted"
            Trivial implementation until we figure out how to search audios.
        """
        self.search_pos = TextInput(
            title="Placeholder search widget",
            width_policy="fit",
            height_policy="fit",
        )
        self.search_neg = TextInput(
            title="Placeholder search widget",
            width_policy="fit",
            height_policy="fit",
        )

    def _layout_widgets(self):
        """
        ???+ note "Define the layout of widgets."
        """
        return column(
            self.data_key_button_group,
            row(*self._dynamic_widgets.values()),
        )

    def activate_search(self, subset, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ help "Help wanted"
            Trivial implementation until we figure out how to search audios.

            [Create an issue](https://github.com/phurwicz/hover/issues/new) if you have an idea :)

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `subset`        | `str`   | the subset to activate search on |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
        """
        super().activate_search()
        self._warn("no search highlight available.")
        return kwargs


class BokehForImage(BokehBaseExplorer):
    """
    ???+ note "`BokehBaseExplorer` with `image` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) image urls in an `image` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "image"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {"label": True, "image": True, "coords": True, "index": True}

    def _setup_search_highlight(self):
        """
        ???+ help "Help wanted"
            Trivial implementation until we figure out how to search images.
        """
        self.search_pos = TextInput(
            title="Placeholder search widget",
            width_policy="fit",
            height_policy="fit",
        )
        self.search_neg = TextInput(
            title="Placeholder search widget",
            width_policy="fit",
            height_policy="fit",
        )

    def _layout_widgets(self):
        """
        ???+ note "Define the layout of widgets."
        """
        return column(
            self.data_key_button_group,
            row(*self._dynamic_widgets.values()),
        )

    def activate_search(self):
        """
        ???+ note "Assign Highlighting callbacks to search results."
        """
        super().activate_search()
        self._warn("no search highlight available.")
