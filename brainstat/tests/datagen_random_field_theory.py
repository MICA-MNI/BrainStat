"""Data generation for random field theory unit tests.


Tests 1-4 test for variance in the input data types.
Tests 5-20 test input of the optional variables. 
Test 21 tests for nonsense variables in the SLM.

"""

import pickle

import numpy as np
from brainspace.mesh.mesh_elements import get_cells, get_edges, get_points
from nilearn import datasets
from sklearn.model_selection import ParameterGrid

from brainstat.context.utils import read_surface_gz
from brainstat.tests.testutil import datadir, generate_slm, save_slm


def generate_random_slm(surf, n_var=1, dfs=None, mask=None, cluster_threshold=0.001):
    """Generates a valid SLM for a surface.

    Parameters
    ----------
    surf : BSPolyData
        Brain surface.
    n_var : int, optional
        slm.k, by default 1.
    dfs : np.array, None, optional
        Effective degrees of freedom, by default None.
    mask : np.array, optional
        Boolean mask, by default None.
    cluster_threshold : float, optional
        Cluster threshold, by default 0.001.

    Returns
    -------
    brainstat.stats.SLM
        SLM object.
    """
    edges = get_edges(surf)
    vertices = get_points(surf)

    n_vertices = vertices.shape[0]
    n_edges = edges.shape[0]

    slm = generate_slm(
        t=np.random.random_sample((1, n_vertices)),
        df=np.random.randint(2, 100),
        k=n_var,
        resl=np.random.random_sample((n_edges, 1)),
        surf=surf,
        dfs=dfs,
        mask=mask,
        cluster_threshold=cluster_threshold,
    )
    return slm


def slm2files(slm, basename, test_num):
    """Converts an SLM to its input/output files

    Parameters
    ----------
    slm : brainstat.stats.SLM
        SLM object.
    basename : str
        Base name for the file.
    test_num : int
        Number of the test.
    """
    D = {}
    D["pval"], D["peak"], D["clus"], D["clusid"] = slm.random_field_theory()
    save_slm(slm, basename, test_num, input=True)
    filename = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    if "_tri" in D:
        D.pop("_tri")
    with open(filename, "wb") as f:
        pickle.dump(D, f, protocol=4)


def generate_test_data():
    ## Fetch global data and settings.
    # Global test settings
    basename = "xstatp"
    test_num = 1
    np.random.seed(0)

    # Fetch surface data.
    pial_fs5 = datasets.fetch_surf_fsaverage()["pial_left"]
    pial_surf = read_surface_gz(pial_fs5)
    n_vertices = get_points(pial_surf).shape[0]

    ## Define the test parameter grids.
    # Variable types to test
    var_types = {
        "t": [np.float64],
        "df": [int, np.uint16],
        "k": [int, np.uint8],
        "resl": [np.float64],
        "tri": [np.int64],
    }
    type_parameter_grid = ParameterGrid(var_types)

    # Optional variable test.
    var_optional = {
        "dfs": [None, np.random.randint(1, 100, (1, n_vertices))],
        "cluster_threshold": [0.1, 2],
        "mask": [None, np.random.rand(n_vertices) > 0.1],
        "k": [1, 3],
    }

    # Nonsense variables to add.
    var_nonsense = ["X", "coef", "SSE", "c", "ef", "sd"]

    ## Generate test data
    # Variable type tests
    for params in type_parameter_grid:
        slm = generate_random_slm(pial_surf)
        for key in list(params.keys()):
            attr = getattr(slm, key)
            setattr(slm, key, params[key](attr))
        slm2files(slm, basename, test_num)
        test_num += 1

    # Additional variable tests.
    additional_parameter_grid = ParameterGrid(var_optional)
    for params in additional_parameter_grid:
        slm = generate_random_slm(pial_surf)
        for key in list(params.keys()):
            setattr(slm, key, params[key])
        slm2files(slm, basename, test_num)
        test_num += 1

    # Nonsense variable tests.
    slm = generate_random_slm(pial_surf)
    slm.dfs = np.random.randint(1, 100, (1, n_vertices))
    slm.mask = np.random.rand(n_vertices) > 0.1
    for key in var_nonsense:
        if getattr(slm, key) is None:
            setattr(
                slm,
                key,
                np.random.rand(np.random.randint(1, 10), np.random.randint(1, 10)),
            )
    slm2files(slm, basename, test_num)
    test_num += 1


if __name__ == "__main__":
    generate_test_data()
