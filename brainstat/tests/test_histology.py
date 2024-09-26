"""Unit tests for the histology module."""
import pytest
import requests
import numpy as np

from brainstat._utils import read_data_fetcher_json
from brainstat.context.histology import read_histology_profile
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


def test_histology_profiles_is_ndarray():
    histology_profiles = read_histology_profile(template="fslr32k")
    
    # Assert that histology_profiles is an ndarray
    assert isinstance(histology_profiles, np.ndarray), "histology_profiles should be a NumPy ndarray"
