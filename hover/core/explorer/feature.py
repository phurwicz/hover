"""
???+ note "Intermediate classes based on the main feature."
"""
import re
from bokeh.models import TextInput
from bokeh.layouts import column, row
from hover.core.local_config import blank_callback_on_change as blank
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

    def _setup_search_widgets(self):
        """
        ???+ note "Create positive/negative text search boxes."
        """
        common_kwargs = dict(width_policy="fit", height_policy="fit")
        pos_title, neg_title = "Text contains (python regex):", "Text does not contain:"
        self.search_pos = TextInput(title=pos_title, **common_kwargs)
        self.search_neg = TextInput(title=neg_title, **common_kwargs)

        def search_base_response(attr, old, new):
            for _subset in self.sources.keys():
                _func = self._dynamic_callbacks["search_response"].get(_subset, blank)
                _func(attr, old, new)
            return

        self.search_pos.on_change("value", search_base_response)
        self.search_neg.on_change("value", search_base_response)

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

    def _setup_search_widgets(self):
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

    def activate_search(self):
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
        self._warn("no search available.")


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

    def _setup_search_widgets(self):
        """
        ???+ note "Create similarity search widgets."
        """
        self.search_sim = TextInput(
            title="Image similarity search form URL",
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
        self._warn("no search available.")
