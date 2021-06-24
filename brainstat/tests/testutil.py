"""Utilities for running tests and test data generation."""
import os
import numpy as np
import pickle
import vtk
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect
from brainspace.vtk_interface import wrap_vtk
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainspace.mesh.mesh_elements import get_cells, get_points


def datadir(file):
    topdir = os.path.dirname(__file__)
    topdir = os.path.join(topdir, "../../")
    topdir = os.path.abspath(topdir)
    return os.path.join(topdir, "extern/test-data", file)


def generate_slm(**kwargs):
    """Generates a SLM with the given attributes
    Parameters
    ----------
    All attributes of SLM can be provided as keyword arguments.
    Returns
    -------
    brainstat.stats.SLM.SLM
        SLM object.
    """
    slm = SLM(FixedEffect(1), 1)
    for key, value in kwargs.items():
        setattr(slm, key, value)
    return slm


def copy_slm(slm):
    """Copies an SLM object.
    Parameters
    ----------
    slm : brainstat.stats.SLM.SLM
        SLM object.
    Returns
    -------
    brainstat.stats.SLM.SLM
        SLM object.
    """
    slm_out = SLM(FixedEffect(1), 1)
    for key in slm.__dict__:
        setattr(slm_out, key, getattr(slm, key))
    return slm_out


def save_slm(slm, basename, testnum, input=True):
    """Saves an SLM object containing test data.
    Parameters
    ----------
    slm : brainstat.stats.SLM.SLM
        SLM object.
    basename : str
        Name for the tested function.
    testnum: int
        Test number.
    input: boolean, optional
        If True, appends _IN to filename. If false appends _OUT.
    Returns
    -------
    brainstat.stats.SLM.SLM
        SLM object.
    """
    D = slm2dict(slm)
    # Custom classes won't support MATLAB conversion.
    D.pop("surf", None)
    D.pop("model", None)
    if input:
        stage = "IN"
    else:
        stage = "OUT"
    filename = datadir(basename + "_" + f"{testnum:02d}" + "_" + stage + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(D, f, protocol=4)


def slm2dict(slm):
    """Converts an SLM to a dictionary.
    Parameters
    ----------
    slm : brainstat.stats.SLM.SLM
        SLM object.
    Returns
    -------
    dict
        Dictionary with keys equivalent to the attributes of the slm.
    """
    D = {}
    for key in slm.__dict__:
        if getattr(slm, key) is not None:
            D[key] = getattr(slm, key)
    return D


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
    if "_surf" in D and isinstance(D["_surf"], BSPolyData):
        D["surf"] = {
            "tri": np.array(get_cells(D["_surf"])),
            "coord": np.array(get_points(D["_surf"])).T,
        }
        D.pop("_surf")
        D.pop("_tri")

    filename = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(filename, "wb") as f:
        pickle.dump(D, f, protocol=4)


def save_input_dict(params, basename, test_num):
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

    if isinstance(params["surf"], BSPolyData):
        params["surf"] = {
            "tri": np.array(get_cells(params["surf"])) + 1,
            "coord": np.array(get_points(params["surf"])).T,
        }

    with open(filename, "wb") as f:
        pickle.dump(params, f, protocol=4)


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


def generate_random_data_model(
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


def array2effect(A, n_random=0):
    fixed_effects = FixedEffect(1, "intercept") + FixedEffect(A[:, n_random:])
    if n_random != 0:
        mixed_effects = (
            MixedEffect(
                A[:, :n_random],
                name_ran=["f" + str(x) for x in range(n_random)],
            )
            + MixedEffect(1, "identity")
        )
        return fixed_effects + mixed_effects
    else:
        return fixed_effects
