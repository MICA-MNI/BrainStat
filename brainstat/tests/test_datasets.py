""" Tests for brainstat.datasets module. """
import numpy as np
import pytest
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat.datasets.base import (
    _valid_parcellations,
    fetch_parcellation,
    fetch_template_surface,
)

parametrize = pytest.mark.parametrize


@parametrize(
    "template",
    ["fslr32k", "fsaverage", "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6"],
)
def test_load_surfaces(template):
    """Test loading surface templates. For now this only tests whether it runs
    without error and whether the output type is correct."""
    surface = fetch_template_surface(template, join=True)
    assert isinstance(surface, BSPolyData)

    surface_lh, surface_rh = fetch_template_surface(template, join=False)

    assert isinstance(surface_lh, BSPolyData)
    assert isinstance(surface_rh, BSPolyData)


valid_n_regions = {k: v["n_regions"] for k, v in _valid_parcellations().items()}
valid_surfaces = {k: v["surfaces"] for k, v in _valid_parcellations().items()}


@parametrize("atlas", valid_n_regions.keys())
@parametrize("template", ["fsaverage", "fsaverage5", "fsaverage6", "fslr32k"])
def test_load_parcels(atlas, template):
    """Test loading surface parcels."""
    if template not in valid_surfaces[atlas]:
        pytest.skip(f"{atlas} does not exist on {template}")

    for n_regions in valid_n_regions[atlas]:
        parcels = fetch_parcellation(
            template, atlas, n_regions, join=True, seven_networks=True
        )
        assert isinstance(parcels, np.ndarray)

        parcels_lh, parcels_rh = fetch_parcellation(
            template, atlas, n_regions, join=False
        )
        assert isinstance(parcels_lh, np.ndarray)
        assert isinstance(parcels_rh, np.ndarray)

        assert parcels.size == parcels_lh.size + parcels_rh.size

        if atlas == "schaefer":
            parcels = fetch_parcellation(
                template, atlas, n_regions, join=True, seven_networks=False
            )
            assert isinstance(parcels, np.ndarray)
