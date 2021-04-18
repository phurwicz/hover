"""Mini-functions that do not belong elsewhere."""
from datetime import datetime
from abc import ABC, abstractmethod


def current_time(template="%Y%m%d %H:%M:%S"):
    return datetime.now().strftime(template)


class BaseUnionFind(ABC):
    """
    ???+ note "Data attached to union-find."
    """

    def __init__(self, data):
        self._data = data
        self._parent = None
        self._count = 1

    def __repr__(self):
        return self.data.__repr__()

    @property
    def count(self):
        if self.parent is None:
            return self._count
        return self.find().count

    @count.setter
    def count(self, count):
        self._count = count

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, other):
        assert isinstance(other, BaseUnionFind)
        self._parent = other

    def find(self):
        if self.parent:
            self.parent = self.parent.find()
            return self.parent
        return self

    @abstractmethod
    def union(self, other):
        pass


class NodeUnionFind(BaseUnionFind):
    """
    ???+ note "Each node keeps its own data."
    """

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def union(self, other):
        root = self.find()
        other_root = other.find()
        if root is other_root:
            return

        # merge the smaller trees into the larger
        if root.count < other_root.count:
            other_root.count += root.count
            root.parent = other_root
        else:
            root.count += other_root.count
            other_root.parent = root


class RootUnionFind(BaseUnionFind):
    """
    ???+ note "Union always uses left as root. Each node looks up its root for data."
    """

    @property
    def data(self):
        root = self.find()
        if self is root:
            return self._data
        return root.data

    @data.setter
    def data(self, data):
        root = self.find()
        if self is root:
            self._data = data
        root._data = data

    def union(self, other):
        root = self.find()
        other_root = other.find()

        # clear the data on the other root
        other_root.data = None
        root.count += other_root.count
        other_root.parent = root
