import pytest
from hover.core.explorer.functionality import (
    BokehDataAnnotator,
    BokehDataFinder,
    BokehSoftLabelExplorer,
    BokehSnorkelExplorer,
)
from hover.recipes.subroutine import (
    standard_annotator,
    standard_finder,
    standard_snorkel,
    standard_softlabel,
)


@pytest.mark.lite
def test_autobuild_explorer(
    example_text_dataset,
    example_image_dataset,
    example_audio_dataset,
):
    for dataset in [
        example_text_dataset,
        example_image_dataset,
        example_audio_dataset,
    ]:
        dataset = dataset.copy()

        annotator = standard_annotator(dataset)
        assert isinstance(annotator, BokehDataAnnotator)

        finder = standard_finder(dataset)
        assert isinstance(finder, BokehDataFinder)

        softlabel = standard_softlabel(dataset)
        assert isinstance(softlabel, BokehSoftLabelExplorer)

        snorkel = standard_snorkel(dataset)
        assert isinstance(snorkel, BokehSnorkelExplorer)
