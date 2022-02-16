from hover.core.local_config import (
    embedding_field,
    is_embedding_field,
)
import pytest


@pytest.mark.lite
def test_embedding_field():
    for i in range(2, 10):
        for j in range(i):
            assert is_embedding_field(embedding_field(i, j))
