"""
Corresponds to the `hover.core.explorer.feature` module.
For mechanisms that are invariant across `hover.core.explorer.functionality`.
"""

import pytest
import math
from hover.recipes.subroutine import get_explorer_class
from .local_helper import (
    FUNCTIONALITY_TO_SPECIAL_ARGS,
    subroutine_search_source_response,
)

MAIN_FUNCTIONALITIES = list(FUNCTIONALITY_TO_SPECIAL_ARGS.keys())


def subroutine_searchable_explorer(dataset, functionality, feature):
    explorer_cls = get_explorer_class(functionality, feature)
    subset_mapping = explorer_cls.DEFAULT_SUBSET_MAPPING.copy()
    special_args = FUNCTIONALITY_TO_SPECIAL_ARGS[functionality]
    explorer = explorer_cls.from_dataset(dataset, subset_mapping, *special_args)
    explorer.activate_search()
    return explorer


@pytest.mark.core
class TestBokehForText:
    @staticmethod
    @pytest.mark.lite
    def test_search(example_text_dataset):
        for _functionality in MAIN_FUNCTIONALITIES:
            _explorer = subroutine_searchable_explorer(
                example_text_dataset,
                _functionality,
                "text",
            )

            def search_a():
                _explorer.search_pos.value = r"a"

            def desearch_a():
                _explorer.search_neg.value = r"a"

            subroutine_search_source_response(
                _explorer,
                [search_a, desearch_a],
            )


@pytest.mark.core
class TestBokehForImage:
    @staticmethod
    @pytest.mark.lite
    def test_search(example_image_dataset):
        for _functionality in MAIN_FUNCTIONALITIES:
            _explorer = subroutine_searchable_explorer(
                example_image_dataset,
                _functionality,
                "image",
            )

            def enter_first_image():
                _explorer.search_sim.value = _explorer.dfs["raw"]["image"][0]

            def enter_second_image():
                _explorer.search_sim.value = _explorer.dfs["raw"]["image"][1]

            subroutine_search_source_response(
                _explorer,
                [enter_first_image, enter_second_image],
            )


@pytest.mark.core
class TestBokehForAudio:
    @staticmethod
    @pytest.mark.lite
    def test_search(example_audio_dataset):
        for _functionality in MAIN_FUNCTIONALITIES:
            _explorer = subroutine_searchable_explorer(
                example_audio_dataset,
                _functionality,
                "audio",
            )

            def enter_first_audio():
                _explorer.search_sim.value = _explorer.dfs["raw"]["audio"][0]

            def alter_sim_thresh():
                shifted = _explorer.search_threshold.value + 0.5
                _explorer.search_threshold.value = shifted - math.floor(shifted)

            subroutine_search_source_response(
                _explorer,
                [enter_first_audio, alter_sim_thresh],
            )
