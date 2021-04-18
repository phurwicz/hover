"""
???+ note "Intermediate classes based on the main feature."
"""
import re
from bokeh.models import TextInput
from bokeh.layouts import column, row
from .base import BokehBaseExplorer
from .local_config import SEARCH_SCORE_FIELD


class BokehForText(BokehBaseExplorer):
    """
    ???+ note "`BokehBaseExplorer` with `text` (`str`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) text data in a `text` column

        Does not assume:

        - what the explorer serves to do.
    """

    MANDATORY_COLUMNS = ["text", "label", "x", "y"]
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

    def activate_search(self, subset, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ note "Enables string/regex search-and-highlight mechanism."
            Modifies the plotting source in-place.
            Using a JS callback (instead of Python) so that it also works in standalone HTML.

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `subset`        | `str`   | the subset to activate search on |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
        """
        assert isinstance(kwargs, dict)
        updated_kwargs = kwargs.copy()

        param_key, param_pos, param_neg, param_default = altered_param
        num_points = len(self.sources[subset].data["text"])
        self.sources[subset].add([param_default] * num_points, f"{param_key}")
        self._extra_source_cols[subset][param_key] = param_default

        updated_kwargs[param_key] = param_key

        def search_response(attr, old, new):
            pos_regex, neg_regex = self.search_pos.value, self.search_neg.value

            def regex_score(text):
                score = 0
                if len(pos_regex) > 0:
                    score += 1 if re.search(pos_regex, text) else -2
                if len(neg_regex) > 0:
                    score += -2 if re.search(neg_regex, text) else 1
                return score

            def score_to_param(score):
                if score > 0:
                    return param_pos
                elif score == 0:
                    return param_default
                else:
                    return param_neg

            patch_slice = slice(len(self.sources[subset].data["text"]))
            search_scores = list(map(regex_score, self.sources[subset].data["text"]))
            search_params = list(map(score_to_param, search_scores))
            self.sources[subset].patch(
                {SEARCH_SCORE_FIELD: [(patch_slice, search_scores)]}
            )
            self.sources[subset].patch({param_key: [(patch_slice, search_params)]})
            return

        # js_callback = CustomJS(
        #    args={
        #        "source": self.sources[subset],
        #        "key_pos": self.search_pos,
        #        "key_neg": self.search_neg,
        #        "param_pos": param_pos,
        #        "param_neg": param_neg,
        #        "param_default": param_default,
        #    },
        #    code=f"""
        #    const data = source.data;
        #    const text = data['text'];
        #    var highlight_arr = data['{param_key}'];
        #    var score_arr = data['{SEARCH_SCORE_FIELD}'];
        #    """
        #    + """
        #    var search_pos = key_pos.value;
        #    var search_neg = key_neg.value;
        #    var valid_pos = (search_pos.length > 0);
        #    var valid_neg = (search_neg.length > 0);
        #
        #    function searchScore(candidate)
        #    {
        #        var score = 0;
        #        if (valid_pos) {
        #            if (candidate.search(search_pos) >= 0) {
        #                score += 1;
        #            } else {
        #                score -= 2;
        #            }
        #        };
        #        if (valid_neg) {
        #            if (candidate.search(search_neg) < 0) {
        #                score += 1;
        #            } else {
        #                score -= 2;
        #            }
        #        };
        #        return score;
        #    }
        #
        #    function scoreToAttr(score)
        #    {
        #        // return attribute
        #        if (score > 0) {
        #            return param_pos;
        #        } else if (score < 0) {
        #            return param_neg;
        #        } else {return param_default;}
        #    }
        #
        #    function toRegex(search_key) {
        #        var match = search_key.match(new RegExp('^/(.*?)/([gimy]*)$'));
        #        if (match) {
        #            return new RegExp(match[1], match[2]);
        #        } else {
        #            return search_key;
        #        }
        #    }
        #
        #    // convert search input to regex
        #    if (valid_pos) {search_pos = toRegex(search_pos);}
        #    if (valid_neg) {search_neg = toRegex(search_neg);}
        #
        #    // search, store scores, and set highlight
        #    for (var i = 0; i < highlight_arr.length; i++) {
        #        var score = searchScore(text[i]);
        #        score_arr[i] = score;
        #        highlight_arr[i] = scoreToAttr(score);
        #    }
        #
        #    source.change.emit()
        #    """,
        # )

        # assign dynamic callback
        self._dynamic_callbacks["search_response"][subset] = search_response
        return updated_kwargs


class BokehForAudio(BokehBaseExplorer):
    """
    ???+ note "`BokehBaseExplorer` with `audio` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) audio urls in an `audio` column

        Does not assume:

        - what the explorer serves to do.
    """

    MANDATORY_COLUMNS = ["audio", "label", "x", "y"]
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

    MANDATORY_COLUMNS = ["image", "label", "x", "y"]
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

    def activate_search(self, subset, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ help "Help wanted"
            Trivial implementation until we figure out how to search images.

            [Create an issue](https://github.com/phurwicz/hover/issues/new) if you have an idea :)

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `subset`        | `str`   | the subset to activate search on |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
        """
        self._warn("no search highlight available.")
        return kwargs
