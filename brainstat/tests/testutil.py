"""Utilities for tests and test data generation."""
import os
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term
import pickle


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
    slm = SLM(Term(1), 1)
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
    slm_out = SLM(Term(1), 1)
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
