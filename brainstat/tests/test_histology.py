"""Unit tests for the histology module."""
import pytest
import requests

from brainstat._utils import read_data_fetcher_json

parametrize = pytest.mark.parametrize
json = read_data_fetcher_json()


@parametrize("template", list(json["bigbrain_profiles"].keys()))
def test_urls(template):
    """Tests whether the histology files can be downloaded.

    Parameters
    ----------
    template : list
        Template names.
    """
    r = requests.head(json["bigbrain_profiles"][template]["url"])
    assert r.status_code == 200
