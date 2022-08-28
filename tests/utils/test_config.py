import pytest
from hover.utils.config import (
    auto_interpret,
    LockableConfig,
)


@pytest.mark.lite
@pytest.mark.parametrize(
    "inp,out",
    [
        ("foo", "foo"),
        ("Foo", "Foo"),
        ("1", 1),
        ("-3", -3),
        ("-3.99", -3.99),
        ("-3.99.0", "-3.99.0"),
        ("true", True),
        ("off", False),
        ("False", False),
        ("-1-2", "-1-2"),
        ([], []),
        ({1}, {1}),
        (None, None),
    ],
)
def test_auto_interpret(inp, out):
    assert auto_interpret(inp) == out


@pytest.mark.lite
class TestLockableConfig:
    @staticmethod
    def test_comprehensive():
        config = LockableConfig({"foo": 1})
        # writing to an unread key-value should work
        config["foo"] = 2

        # reading a key locks its value
        _ = config["foo"]

        # writing to a locked key-value should fail
        try:
            config["foo"] = 3
            raise RuntimeError("Expected value to be locked.")
        except AssertionError:
            pass

        # write to a new key-value should work
        config.update({"goo": 1, "bar": 0})

        # the update method should not bypass the lock
        try:
            _ = config["goo"]
            config.update({"goo": 1, "bar": 1})
            raise RuntimeError("Expected value to be locked.")
        except AssertionError:
            # the entire update is expected to fail, not just locked parts
            # note, for testing purpose, not to directly access the value
            assert config._data["bar"] == 0
            # updating unlocked parts should still work
            config.update({"bar": 1})
            assert config._data["bar"] == 1
