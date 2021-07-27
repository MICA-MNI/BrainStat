""" Tests for brainstat.datasets module. """
import numpy as np
import pytest
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat.datasets import fetch_parcellation, fetch_template_surface

parametrize = pytest.mark.parametrize
valid_n_regions = {
    "schaefer": (100, 200, 300, 400, 600, 800, 1000),
    "cammoun": (33, 60, 125, 250, 500),
}


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


@parametrize("atlas", valid_n_regions.keys())
@parametrize("template", ["fsaverage", "fsaverage5", "fsaverage6", "fslr32k"])
def test_load_parcels(atlas, template):
    """Test loading surface parcels."""
    for n_regions in valid_n_regions[atlas]:
        parcels = fetch_parcellation(atlas, n_regions, template=template, join=True)
        assert isinstance(parcels, np.ndarray)

        parcels_lh, parcels_rh = fetch_parcellation(
            atlas, n_regions, template=template, join=False
        )
        assert isinstance(parcels_lh, np.ndarray)
        assert isinstance(parcels_rh, np.ndarray)

        assert parcels.size == parcels_lh.size + parcels_rh.size

        if atlas == "schaefer":
            parcels = fetch_parcellation(
                atlas, n_regions, template=template, join=True, seven_networks=False
            )
            assert isinstance(parcels, np.ndarray)
