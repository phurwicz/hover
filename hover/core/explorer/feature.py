"""
???+ note "Intermediate classes based on the main feature."
"""
import re
import numpy as np
from functools import lru_cache
from bokeh.models import TextInput
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

    def activate_search(self):
        """
        ???+ note "Bind search response callbacks to widgets."
        """
        super().activate_search()
        self.search_pos.on_change("value", self.search_base_response)
        self.search_neg.on_change("value", self.search_base_response)

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


class BokehForVector(BokehBaseExplorer):
    """
    ???+ note "A layer of abstraction for `BokehBaseExplorer` subclasses whose feature-type-specific mechanisms work the same way through vectors."
    """

    def _setup_search_widgets(self):
        """
        ???+ note "Create similarity search widgets."
        """
        self.search_sim = TextInput(
            title=f"{self.__class__.PRMARY_FEATURE} similarity search",
            width_policy="fit",
            height_policy="fit",
        )

    def _setup_search_resource(self):
        """
        ???+ note "Create nearest neighbors data structure."
        """
        # determine cache size for normalized vectorizer
        num_points = sum([_df.shape[0] for _df in self.dfs])
        cache_size = min(num_points, int(1e5))

        # find vectorizer
        assert hasattr(self, "linked_dataset"), "need linked_dataset for its vectorizer"
        found_vectorizers = self.linked_dataset.vectorizer_lookup.values()
        assert len(found_vectorizers) > 0, "dataset has no known vectorizer"

        raw_vectorizer = list(found_vectorizers)[0]

        # gain speed up by caching and normalization
        @lru_cache(maxsize=cache_size)
        def normalized_vectorizer(feature):
            vec = raw_vectorizer(feature)
            norm = np.linalg.norm(vec)
            return vec / (np.sqrt(norm) + 1e-16)

        self._dynamic_resources["normalized_vectorizer"] = normalized_vectorizer

    def activate_search(self):
        """
        ???+ note "Bind search response callbacks to widgets."
        """
        self._setup_search_resources()
        super().activate_search()
        self.search_sim.on_change("value", self.search_base_response)

    def _get_search_score_function(self):
        """
        ???+ note "Dynamically create a single-argument scoring function."
        """
        vectorizer = self._dynamic_resources["normalized_vectorizer"]
        img_query = self.search_sim.value
        vec_query = vectorizer(img_query)
        assert isinstance(
            vec_query, np.array
        ), f"vector should be np.array, got {type(vec_query)}"

        def cosine_based_score(img_doc, pos_thresh=0.9, neg_thresh=0.7):
            vec_doc = vectorizer(img_doc)
            query_doc_sim = np.dot(vec_query, vec_doc)

            if query_doc_sim > pos_thresh:
                return 1
            elif query_doc_sim < neg_thresh:
                return -1
            else:
                return 0

        return cosine_based_score


class BokehForAudio(BokehForVector):
    """
    ???+ note "`BokehForVector` with `audio` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) audio urls in an `audio` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "audio"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {"label": True, "audio": True, "coords": True, "index": True}


class BokehForImage(BokehForVector):
    """
    ???+ note "`BokehForVector` with `image` (path like `"http://"` or `"file:///"`) as the main feature."
        Assumes on top of its parent class:

        - in supplied dataframes
          - (always) image urls in an `image` column

        Does not assume:

        - what the explorer serves to do.
    """

    PRIMARY_FEATURE = "image"
    MANDATORY_COLUMNS = [PRIMARY_FEATURE, "label"]
    TOOLTIP_KWARGS = {"label": True, "image": True, "coords": True, "index": True}
