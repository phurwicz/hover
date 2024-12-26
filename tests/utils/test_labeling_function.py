from hover.utils.labeling_function import labeling_function
import pytest


@pytest.mark.lite
def test_labeling_function(example_raw_df):
    def original(row):
        return "long" if len(row["text"]) > 5 else "short"

    targets = ["long", "short"]
    one_row = example_raw_df.get_row_as_dict(0)

    original_lf = labeling_function(
        targets=targets,
    )(original)
    renamed_lf = labeling_function(
        targets=targets,
        name="override",
    )(original)

    # check output type
    assert isinstance(original_lf(one_row), str)
    assert isinstance(renamed_lf(one_row), str)

    # check output is within targets
    assert original_lf(one_row) in targets
    assert renamed_lf(one_row) in targets

    # check function name
    assert original_lf.name == "original"
    assert renamed_lf.name == "override"
