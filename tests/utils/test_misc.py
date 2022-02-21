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

    for _l, _r, _nodes, _finds, _counts in [
        (
            0,
            1,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 2, 3, 4, 5, 6, 7],
            [2, 2, 1, 1, 1, 1, 1, 1],
        ),
        (
            1,
            2,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 3, 4, 5, 6, 7],
            [3, 3, 3, 1, 1, 1, 1, 1],
        ),
        (
            0,
            2,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 3, 4, 5, 6, 7],
            [3, 3, 3, 1, 1, 1, 1, 1],
        ),
        (
            3,
            4,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 3, 3, 5, 6, 7],
            [3, 3, 3, 2, 2, 1, 1, 1],
        ),
        (
            4,
            2,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 0, 0, 5, 6, 7],
            [5, 5, 5, 5, 5, 1, 1, 1],
        ),
    ]:
        arr[_l].union(arr[_r])
        check_unionfind(arr, _nodes, _finds, _counts)

    # test data assignment
    arr[0].data = 8
    check_unionfind(
        arr,
        [8, 1, 2, 3, 4, 5, 6, 7],
        [8, 8, 8, 8, 8, 5, 6, 7],
        [5, 5, 5, 5, 5, 1, 1, 1],
    )


@pytest.mark.lite
def test_rootunionfind():
    arr = [RootUnionFind(i) for i in range(8)]

    for _l, _r, _nodes, _finds, _counts in [
        (
            0,
            1,
            [0, 0, 2, 3, 4, 5, 6, 7],
            [0, 0, 2, 3, 4, 5, 6, 7],
            [2, 2, 1, 1, 1, 1, 1, 1],
        ),
        (
            1,
            2,
            [0, 0, 0, 3, 4, 5, 6, 7],
            [0, 0, 0, 3, 4, 5, 6, 7],
            [3, 3, 3, 1, 1, 1, 1, 1],
        ),
        (
            3,
            4,
            [0, 0, 0, 3, 3, 5, 6, 7],
            [0, 0, 0, 3, 3, 5, 6, 7],
            [3, 3, 3, 2, 2, 1, 1, 1],
        ),
        (
            4,
            2,
            [3, 3, 3, 3, 3, 5, 6, 7],
            [3, 3, 3, 3, 3, 5, 6, 7],
            [5, 5, 5, 5, 5, 1, 1, 1],
        ),
    ]:
        arr[_l].union(arr[_r])
        check_unionfind(arr, _nodes, _finds, _counts)
