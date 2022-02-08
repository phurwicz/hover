import pytest
from hover.utils.denoising import (
    identity_adjacency,
    cyclic_adjacency,
    cyclic_except_last,
    accuracy_priority,
    disagreement_priority,
)
from collections import defaultdict


def array_to_index_keyed_dict(arr):
    data_dict = defaultdict(dict)
    for i, _row in enumerate(arr):
        for j, _value in enumerate(_row):
            data_dict[i][j] = _value

    return data_dict


@pytest.fixture
def example_info_dict():
    return {
        "accuracy": [0.613, 0.618, 0.623, 0.605],
        "disagreement_rate": array_to_index_keyed_dict(
            [
                [0.0, 0.11, 0.12, 0.09],
                [0.11, 0.0, 0.13, 0.14],
                [0.12, 0.13, 0.0, 0.11],
                [0.09, 0.14, 0.11, 0.0],
            ]
        ),
    }


@pytest.mark.lite
def test_identity_adjacency(example_info_dict):
    adjacency_list = identity_adjacency(example_info_dict)
    for i, _neighbors in enumerate(adjacency_list):
        assert _neighbors == [i], "Expected identity adjacency"


@pytest.mark.lite
def test_cyclic_adjacency(example_info_dict):
    adjacency_list = cyclic_adjacency(example_info_dict)
    num_nodes = len(example_info_dict["accuracy"])
    for i, _neighbors in enumerate(adjacency_list):
        assert _neighbors == [(i + 1) % num_nodes], "Expected cyclic adjacency"


@pytest.mark.lite
def test_cyclic_except_last(example_info_dict):
    adjacency_list = cyclic_except_last(example_info_dict)
    num_nodes = len(example_info_dict["accuracy"])
    for i, _neighbors in enumerate(adjacency_list):
        if i != num_nodes - 1:
            assert _neighbors == [
                (i + 1) % (num_nodes - 1)
            ], f"Expected cyclic on node {i}"
        else:
            assert _neighbors == [i], f"Expected identity on node {i}"


@pytest.mark.lite
def test_accuracy_priority(example_info_dict):
    adjacency_list = accuracy_priority(example_info_dict)
    assert adjacency_list == [
        [2],
        [2],
        [1],
        [2],
    ], "Expected every node to point to the one with best accuracy except itself"


@pytest.mark.lite
def test_disagreement_priority(example_info_dict):
    adjacency_list = disagreement_priority(example_info_dict)
    assert adjacency_list == [
        [2],
        [3],
        [1],
        [1],
    ], "Expected every node to point to the one with most disagreement from itself"
