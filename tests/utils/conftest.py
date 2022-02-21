import pytest
from bokeh.plotting import figure


@pytest.fixture
def dummy_working_recipe():
    def recipe(*args, **kwargs):
        return figure()

    return recipe


@pytest.fixture
def dummy_broken_recipe():
    def recipe(*args, **kwargs):
        assert False

    return recipe
