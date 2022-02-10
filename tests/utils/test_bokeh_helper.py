from hover.utils.bokeh_helper import binder_proxy_app_url, remote_jupyter_proxy_url
from urllib.parse import urlparse
import pytest
import os


@pytest.mark.lite
def test_binder_proxy_app_url():
    """
    Not a full test, rather just validating urls.
    """
    os.environ["JUPYTERHUB_SERVICE_PREFIX"] = "hover-binder"
    url = binder_proxy_app_url("simple-annotator", port=5007)
    _ = urlparse(url)
    os.environ.pop("JUPYTERHUB_SERVICE_PREFIX")


@pytest.mark.lite
def test_remote_jupyter_proxy_url():
    """
    Not a full test, rather just validating urls.
    """
    url = remote_jupyter_proxy_url(8888)
    _ = urlparse(url)
