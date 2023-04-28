from hover.utils.typecheck import TypedValueDict
from collections import defaultdict


class TestTypedValueDict:
    def test_basic(self):
        tdict = TypedValueDict(int)
        tdict["key1"] = 1
        assert tdict["key1"] == 1

        tdict.update({"key2": 2, "key3": 3})
        assert tdict["key2"] == 2
        assert tdict["key3"] == 3

        try:
            tdict["key4"] = "4"
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            pass

    def test_subclass(self):
        tdict = TypedValueDict(dict)
        tdict["key1"] = {"foo": "bar"}
        assert tdict["key1"] == {"foo": "bar"}

        ddict = defaultdict(str)
        tdict.update({"key2": ddict})
        assert tdict["key2"] is ddict
