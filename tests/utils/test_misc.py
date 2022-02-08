from hover.utils.misc import current_time, NodeUnionFind, RootUnionFind
import pytest


@pytest.mark.lite
def test_current_time():
    timestamp = current_time()
    assert isinstance(timestamp, str)


def node_data_from_uf_array(arr):
    """Subroutine for testing utility."""
    return [_node.data for _node in arr]


def find_data_from_uf_array(arr):
    """Subroutine for testing utility."""
    return [_node.find().data for _node in arr]


def counts_from_uf_array(arr):
    """Subroutine for testing utility."""
    return [_node.count for _node in arr]


def check_unionfind(arr, nodes, finds, counts):
    assert node_data_from_uf_array(arr) == nodes
    assert find_data_from_uf_array(arr) == finds
    assert counts_from_uf_array(arr) == counts


@pytest.mark.lite
def test_nodeunionfind():
    arr = [NodeUnionFind(i) for i in range(8)]
    assert repr(arr[0]) == "0"

    arr[0].union(arr[1])
    check_unionfind(
        arr,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 2, 3, 4, 5, 6, 7],
        [2, 2, 1, 1, 1, 1, 1, 1],
    )

    arr[1].union(arr[2])
    check_unionfind(
        arr,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 0, 3, 4, 5, 6, 7],
        [3, 3, 3, 1, 1, 1, 1, 1],
    )

    arr[3].union(arr[4])
    check_unionfind(
        arr,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 0, 3, 3, 5, 6, 7],
        [3, 3, 3, 2, 2, 1, 1, 1],
    )

    arr[4].union(arr[2])
    check_unionfind(
        arr,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 0, 0, 0, 5, 6, 7],
        [5, 5, 5, 5, 5, 1, 1, 1],
    )


@pytest.mark.lite
def test_rootunionfind():
    arr = [RootUnionFind(i) for i in range(8)]

    arr[0].union(arr[1])
    check_unionfind(
        arr,
        [0, 0, 2, 3, 4, 5, 6, 7],
        [0, 0, 2, 3, 4, 5, 6, 7],
        [2, 2, 1, 1, 1, 1, 1, 1],
    )

    arr[1].union(arr[2])
    check_unionfind(
        arr,
        [0, 0, 0, 3, 4, 5, 6, 7],
        [0, 0, 0, 3, 4, 5, 6, 7],
        [3, 3, 3, 1, 1, 1, 1, 1],
    )

    arr[3].union(arr[4])
    check_unionfind(
        arr,
        [0, 0, 0, 3, 3, 5, 6, 7],
        [0, 0, 0, 3, 3, 5, 6, 7],
        [3, 3, 3, 2, 2, 1, 1, 1],
    )

    arr[4].union(arr[2])
    check_unionfind(
        arr,
        [3, 3, 3, 3, 3, 5, 6, 7],
        [3, 3, 3, 3, 3, 5, 6, 7],
        [5, 5, 5, 5, 5, 1, 1, 1],
    )
