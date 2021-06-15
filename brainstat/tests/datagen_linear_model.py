import numpy as np
import pickle
from brainstat.tests.testutil import datadir, slm2dict
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect
from sklearn.model_selection import ParameterGrid
import vtk
from brainspace.vtk_interface import wrap_vtk
from brainspace.mesh.mesh_elements import get_cells, get_points


def generate_random_test_data(
    n_observations,
    n_vertices,
    n_variates,
    n_predictors,
):
    """Generates random test data.

    Parameters
    ----------
    n_observations : int
        Number of observations.
    n_vertices : int
        Number of vertices.
    n_variates : int
        Number of variates.
    n_predictors : int
        Number of predictors.
    n_random : int
        Number of random effects.

    Returns
    -------
    numpy.ndarray
        Random data.
    numpy.ndarray
        Random model.
    """
    Y = np.random.random_sample((n_observations, n_vertices, n_variates))
    M = np.random.random_sample((n_observations, n_predictors))
    return Y, M


def generate_test_data():
    """Wrapper function for generating test data. 
    """

    np.random.seed(0)
    surface = _generate_sphere()
    parameters = [
        {
            "n_observations": [103],
            "n_vertices": [np.array(get_points(surface)).shape[0]],
            "n_variates": [1, 2, 3],
            "n_predictors": [1, 7],
            "n_random": [0],
            "surf": [None, surface],
        },
        {
            "n_observations": [103],
            "n_vertices": [np.array(get_points(surface)).shape[0]],
            "n_variates": [1],
            "n_predictors": [2, 7],
            "n_random": [1],
            "surf": [None, surface],
        },
    ]

    test_num = 0
    for params in ParameterGrid(parameters):
        test_num += 1
        Y, M = generate_random_test_data(
            params["n_observations"],
            params["n_vertices"],
            params["n_variates"],
            params["n_predictors"],
        )

        save_input({'Y': Y, 'M': M, 'surf': params['surf'], 'n_random': params['n_random']}, 'xlinmod', test_num)

        # Convert M to a true BrainStat model
        fixed_effects = FixedEffect(1, "intercept") + FixedEffect(M[:, params['n_random']:])
        if params['n_random'] != 0:
            mixed_effects = MixedEffect(
                M[:, :params['n_random']], name_ran=["f" + str(x) for x in range(params['n_random'])]
            ) + MixedEffect(1, "identity")
            M = fixed_effects + mixed_effects
        else:
            M = fixed_effects

        slm = SLM(M, FixedEffect(1), params["surf"])
        slm.linear_model(Y)
        slm2files(slm, "xlinmod", test_num)


def _generate_sphere():
    """Generates a vtk sphere of 50 vertices.

    Returns
    -------
    BSPolyData
        Mesh of a sphere.
    """
    s = vtk.vtkSphereSource()
    s.Update()
    return wrap_vtk(s.GetOutput())


def save_input(params, basename, test_num):
    """Saves the input data.

    Parameters
    ----------
    params : dict
        Parameters provided by the parameter grid.
    basename : str
        Tag to save the file with.
    test_num : int
        Number of the test.
    """
    filename = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    if params['surf'] is not None:
        params['surf'] = {'tri': np.array(get_cells(params['surf'])) + 1, 'coord': np.array(get_points(params['surf']))}
    with open(filename, "wb") as f:
        pickle.dump(params, f, protocol=4)


def slm2files(slm, basename, test_num):
    """Converts an SLM to its output files.

    Parameters
    ----------
    slm : brainstat.stats.SLM
        SLM object.
    basename : str
        Base name for the file.
    test_num : int
        Number of the test.
    """
    D = slm2dict(slm)
    D.pop("model")
    D.pop("contrast")
    if "surf" in D and D["surf"] is not None:
        D["surf"] = {"tri": np.array(get_cells(D["surf"])), "coord": np.array(get_points(D["surf"]))}

    filename = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(filename, "wb") as f:
        pickle.dump(D, f, protocol=4)


if __name__ == "__main__":
    generate_test_data()
