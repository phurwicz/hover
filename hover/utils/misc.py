"""Mini-functions that do not belong elsewhere."""
from datetime import datetime


def current_time(template="%Y%m%d %H:%M:%S"):
    return datetime.now().strftime(template)


class UnionFindNode:
    """
    ???+ note "Data attached to union-find."
    """

    def __init__(self, data):
        self.__data = data
        self.__parent = None
        self.__count = 1

    def __repr__(self):
        return self.data.__repr__()

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    @property
    def count(self):
        if self.parent is None:
            return self.__count
        return self.find().count

    @count.setter
    def count(self, count):
        self.__count = count

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, other):
        assert isinstance(other, UnionFindNode)
        self.__parent = other

    def find(self):
        if self.parent:
            self.parent = self.parent.find()
            return self.parent
        return self

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


class Alphabet:
    def __init__(self, value):
        self._value = value

    # getting the values
    @property
    def value(self):
        print("Getting value")
        return self._value

    # setting the values
    @value.setter
    def value(self, value):
        print("Setting value to " + value)
        self._value = value

    # deleting the values
    @value.deleter
    def value(self):
        print("Deleting value")
        del self._value
