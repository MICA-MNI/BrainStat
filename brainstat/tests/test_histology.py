"""Tests for histology."""

import pytest
from brainstat.context.histology import _get_urls
import requests

parametrize = pytest.mark.parametrize
urls = _get_urls()


@parametrize("template", list(urls.keys()))
def test_urls(template):
    """Tests whether the histology files can be downloaded.

    Parameters
    ----------
    template : list
        Template names.
    """
    r = requests.head(urls[template])
    assert r.status_code == 200
