"""
???+ note "Intermediate classes based on the main feature."
"""
from bokeh.models import CustomJS, ColumnDataSource
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

    MANDATORY_COLUMNS = ["text", "label", "x", "y"]
    TOOLTIP_KWARGS = {"label": True, "text": True, "coords": True, "index": True}

    def _setup_search_highlight(self):
        """
        ???+ note "Create positive/negative text search boxes."
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

    def _layout_widgets(self):
        """
        ???+ note "Define the layout of widgets."
        """
        from bokeh.layouts import column

        return column(self.search_pos, self.search_neg, self.data_key_button_group)

    def activate_search(self, source, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ note "Enables string/regex search-and-highlight mechanism."
            Modifies the plotting source in-place.
            Using a JS callback (instead of Python) so that it also works in standalone HTML.

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `source`        | `bool`  | the `ColumnDataSource` to use |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
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
        self._warn("no search highlight available.")

    def _layout_widgets(self):
        """
        ???+ note "Define the layout of widgets."
        """
        return self.data_key_button_group

    def activate_search(self, source, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ help "Help wanted"
            Trivial implementation until we figure out how to search audios.

            [Create an issue](https://github.com/phurwicz/hover/issues/new) if you have an idea :)

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `source`        | `bool`  | the `ColumnDataSource` to use |
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
        self._warn("no search highlight available.")

    def _layout_widgets(self):
        """
        ???+ note "Define the layout of widgets."
        """
        return self.data_key_button_group

    def activate_search(self, source, kwargs, altered_param=("size", 10, 5, 7)):
        """
        ???+ help "Help wanted"
            Trivial implementation until we figure out how to search images.

            [Create an issue](https://github.com/phurwicz/hover/issues/new) if you have an idea :)

            | Param           | Type    | Description                   |
            | :-------------- | :------ | :---------------------------  |
            | `source`        | `bool`  | the `ColumnDataSource` to use |
            | `kwargs`        | `bool`  | kwargs for the plot to add to |
            | `altered_param` | `tuple` | (attribute, positive, negative, default) |
        """
        self._warn("no search highlight available.")
        return kwargs
