import pytest
from hover.recipes.subroutine import (
    standard_annotator,
    standard_finder,
    standard_snorkel,
    standard_softlabel,
)


@pytest.mark.lite
def test_standard_annotator(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    annotator = standard_annotator(dataset)
    assert annotator


@pytest.mark.lite
def test_standard_finder(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    finder = standard_finder(dataset)
    assert finder


@pytest.mark.lite
def test_standard_snorkel(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    snorkel = standard_snorkel(dataset)
    assert snorkel


@pytest.mark.lite
def test_standard_softlabel(mini_supervisable_text_dataset_embedded):
    dataset = mini_supervisable_text_dataset_embedded.copy()
    softlabel = standard_softlabel(dataset)
    assert softlabel
