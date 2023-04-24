from hover.utils.snorkel_helper import labeling_function
import pytest


@pytest.mark.lite
def test_labeling_function(example_raw_df):
    def original(row):
        return "long" if len(row["text"]) > 5 else "short"

    targets = ["long", "short"]
    one_row = example_raw_df.row(0)

    # create LF with pre-determined label encodings
    label_encoder = {t: i for i, t in enumerate(targets)}
    preencoded = labeling_function(
        targets=targets,
        label_encoder=label_encoder,
        name="pre-encoded",
    )(original)

    assert isinstance(preencoded(one_row), str)
    assert isinstance(preencoded.snorkel(one_row), int)

    # create LF with undetermined label encodings
    unencoded = labeling_function(
        targets=targets,
        label_encoder=None,
        name="unencoded",
    )(original)

    assert isinstance(unencoded(one_row), str)
    assert unencoded.snorkel is None
