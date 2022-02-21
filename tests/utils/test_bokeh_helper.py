import pytest
from urllib.parse import urlparse
from hover.utils.bokeh_helper import (
    servable,
    binder_proxy_app_url,
    remote_jupyter_proxy_url,
)
from tests.recipes.local_helper import execute_handle_function


@pytest.mark.lite
def test_binder_proxy_app_url():
    """
    The function being tested is only intended for Binder.
    """
    url = binder_proxy_app_url("simple-annotator", port=5007)
    _ = urlparse(url)


@pytest.mark.lite
def test_remote_jupyter_proxy_url():
    """
    Not a full test, rather just validating urls.
    """
    for port in [8888, None]:
        url = remote_jupyter_proxy_url(8888)
        _ = urlparse(url)


@pytest.mark.lite
def test_servable_wrapper(dummy_working_recipe, dummy_broken_recipe):
    try:
        dummy_broken_recipe()
        pytest.fail("The dummy broken recipe above should have raised an exception.")
    except AssertionError:
        pass

    for recipe in [dummy_working_recipe, dummy_broken_recipe]:
        handle = servable()(recipe)
        execute_handle_function(handle)
