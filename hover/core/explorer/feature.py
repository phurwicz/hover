"""
???+ note "Intermediate classes based on the main feature."
"""
import re
import hover
import numpy as np
from functools import lru_cache
from bokeh.models import TextInput, Slider
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
    TOOLTIP_KWARGS = {
        "label": {"label": "Label"},
        "text": {"text": "Text"},
        "coords": True,
        "index": True,
    }

    def _setup_search_widgets(self):
        """
        ???+ note "Create positive/negative text search boxes."
        """
        common_kwargs = dict(width_policy="fit", height_policy="fit")
        pos_title, neg_title = "Text contains (python regex):", "Text does not contain:"
        self.search_pos = TextInput(title=pos_title, **common_kwargs)
        self.search_neg = TextInput(title=neg_title, **common_kwargs)

    def _search_watch_widgets(self):
        return [self.search_pos, self.search_neg]

    def _validate_search_input(self):
        """
        ???+ note "Text uses regex search, for which any string can be considered valid."
        """
        return True

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


class BokehForUrlToVector(BokehBaseExplorer):
    """
    ???+ note "A layer of abstraction for `BokehBaseExplorer` subclasses whose feature-type-specific mechanisms work the same way through vectors."
    """

    def _setup_search_widgets(self):
        """
        ???+ note "Create similarity search widgets."
        """
        self.search_sim = TextInput(
            title=f"{self.__class__.PRIMARY_FEATURE} similarity search (enter URL)".capitalize(),
            width_policy="fit",
            height_policy="fit",
        )
        self.search_threshold = Slider(
            start=0.0,
            end=1.0,
            value=0.9,
            # fewer steps allowed because refreshing search result can be expensive
            step=0.1,
            title="Similarity threshold",
        )

    def _search_watch_widgets(self):
        return [self.search_sim, self.search_threshold]

    def _subroutine_search_create_callbacks(self):
        """
        ???+ note "Create search callback functions based on feature attributes."
        """
        # determine cache size for normalized vectorizer
        num_points = sum([_df.shape[0] for _df in self.dfs.values()])
        cache_size = min(num_points, int(1e5))

        # find vectorizer
        assert hasattr(self, "linked_dataset"), "need linked_dataset for its vectorizer"
        found_vectorizers = self.linked_dataset.vectorizer_lookup.values()
        assert len(found_vectorizers) > 0, "dataset has no known vectorizer"

        raw_vectorizer = list(found_vectorizers)[0]

        # gain speed up by caching and normalization
        @lru_cache(maxsize=cache_size)
        def normalized_vectorizer(feature):
            try:
                vec = raw_vectorizer(feature)
            except Exception as e:
                self._warn(f"vectorizer crashed: {e}; assigning None as vector.")
                return None
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-16)

        self._dynamic_resources["normalized_vectorizer"] = normalized_vectorizer

        super()._subroutine_search_create_callbacks()

    def _get_search_score_function(self):
        """
        ???+ note "Dynamically create a single-argument scoring function."
        """
        vectorizer = self._dynamic_resources["normalized_vectorizer"]
        url_query = self.search_sim.value
        sim_thresh = self.search_threshold.value
        vec_query = vectorizer(url_query)

        def cosine_based_score(url_doc):
            # edge case: query or doc is invalid for vectorization
            vec_doc = vectorizer(url_doc)
            if vec_query is None or vec_doc is None:
                return 0

            # common case: query and doc are both valid
            query_doc_sim = (np.dot(vec_query, vec_doc) + 1.0) / 2.0
            if query_doc_sim >= sim_thresh:
                return 1
            else:
                return -1

        return cosine_based_score

    def _validate_search_input(self):
        """
        ???+ note "Must be some url pointing to a suffixed file."

            For speed, avoid sending web requests in this validation step.
        """
        from urllib.parse import urlparse
        from pathlib import Path

        url_query = self.search_sim.value
        file_path = Path(urlparse(url_query).path)
        return bool(file_path.suffix)


class BokehForAudio(BokehForUrlToVector):
    """
    ???+ note "`BokehForUrlToVector` with `audio` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) audio urls in an `audio` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "audio"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {
        "label": {"label": "Label"},
        "audio": {"audio": ""},
        "coords": True,
        "index": True,
    }


class BokehForImage(BokehForUrlToVector):
    """
    ???+ note "`BokehForUrlToVector` with `image` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) image urls in an `image` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "image"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {
        "label": {"label": "Label"},
        "image": {"image": hover.config["visual"]["tooltip_img_style"]},
        "coords": True,
        "index": True,
    }
