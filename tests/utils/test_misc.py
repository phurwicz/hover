from hover.utils.misc import current_time, NodeUnionFind, RootUnionFind


def test_current_time():
    timestamp = current_time()
    assert isinstance(timestamp, str)


def test_nodeunionfind():
    arr = [NodeUnionFind(i) for i in range(8)]
    arr[0].union(arr[1])
    arr[1].union(arr[2])
    arr[3].union(arr[4])

    node_data = [_node.data for _node in arr]
    find_data = [_node.find().data for _node in arr]
    count_arr = [_node.count for _node in arr]

    assert node_data == [0, 1, 2, 3, 4, 5, 6, 7]
    assert find_data == [0, 0, 0, 3, 3, 5, 6, 7]
    assert count_arr == [3, 3, 3, 2, 2, 1, 1, 1]

    arr[2].union(arr[4])

    find_data = [_node.find().data for _node in arr]
    assert find_data == [0, 0, 0, 0, 0, 5, 6, 7]


def test_rootunionfind():
    arr = [RootUnionFind(i) for i in range(8)]
    arr[0].union(arr[1])
    arr[1].union(arr[2])
    arr[3].union(arr[4])

    node_data = [_node.data for _node in arr]
    find_data = [_node.find().data for _node in arr]
    count_arr = [_node.count for _node in arr]

    assert node_data == [0, 0, 0, 3, 3, 5, 6, 7]
    assert find_data == [0, 0, 0, 3, 3, 5, 6, 7]
    assert count_arr == [3, 3, 3, 2, 2, 1, 1, 1]

    arr[2].union(arr[4])

    node_data = [_node.data for _node in arr]
    assert node_data == [0, 0, 0, 0, 0, 5, 6, 7]
