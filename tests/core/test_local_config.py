from bokeh.models import (
    TableColumn,
)
from hover.core.local_config import (
    embedding_field,
    is_embedding_field,
    blank_callback_on_change,
    dataset_default_sel_table_columns,
    dataset_default_sel_table_kwargs,
)
import pytest


@pytest.mark.lite
def test_embedding_field():
    for i in range(2, 10):
        for j in range(i):
            assert is_embedding_field(embedding_field(i, j))


@pytest.mark.lite
def test_blank_callback_on_change():
    blank_callback_on_change("value", 0, 1)

    try:
        blank_callback_on_change()
        pytest.fail(
            "Expected blank_callback_on_change to have signature attr, old, new."
        )
    except TypeError:
        pass


@pytest.mark.lite
def test_dataset_default_sel_table_columns():
    for feature in ["text", "image", "audio"]:
        columns = dataset_default_sel_table_columns(feature)
        assert isinstance(columns, list)
        assert isinstance(columns[0], TableColumn)

    try:
        dataset_default_sel_table_columns("invalid_feature")
        pytest.fail("Expected an exception from creating columns on invalid feature.")
    except ValueError:
        pass


@pytest.mark.lite
def test_dataset_default_sel_table_kwargs():
    for feature in ["text", "image", "audio"]:
        kwargs = dataset_default_sel_table_kwargs(feature)
        assert isinstance(kwargs, dict)
        assert kwargs

    try:
        dataset_default_sel_table_kwargs("invalid_feature")
        pytest.fail("Expected an exception from creating kwargs on invalid feature.")
    except ValueError:
        pass
