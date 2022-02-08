from hover.utils.bokeh_helper import remote_jupyter_proxy_url
from urllib.parse import urlparse
import pytest


@pytest.mark.lite
def test_remote_jupyter_proxy_url():
    """
    Not a full test, rather just validating urls.
    """
    url = remote_jupyter_proxy_url(8888)
    _ = urlparse(url)
