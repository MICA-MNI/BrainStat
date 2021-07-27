"""Unit tests of SLM."""
import pickle

import numpy as np
import pytest

from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect
from brainstat.tests.testutil import datadir


def recursive_comparison(X1, X2):
    """Recursively compares lists/dictionaries."""

    if type(X1) != type(X2):
        raise ValueError("Both inputs must be of the same type.")

    if isinstance(X1, dict):
        if len(X1.keys()) != len(X2.keys()):
            raise ValueError("Different number of keys in each dictionary.")
        iterator = zip(X1.values(), X2.values())
    elif isinstance(X1, list):
        if len(X1) != len(X2):
            raise ValueError("Different number of elements in each list.")
        iterator = zip(X1, X2)
    else:
        # Assume not iterable.
        iterator = zip([X1], [X2])

    output = True
    for x, y in iterator:
        if x is None and y is None:
            output = True
        elif isinstance(x, list) or isinstance(x, dict):
            output = recursive_comparison(x, y)
        else:
            output = np.allclose(x, y)
        if not output:
            return output
    return output


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(FixedEffect(1), FixedEffect(1))
    # Data are saved a little differently from the actual input due to compatibility with MATLAB.
    # Data wrangle a bit to bring it back into the Python input format.
    for key in idic.keys():
        if key == "Y":
            # Y is input for slm.fit(), not a property.
            continue
        if key == "model":
            # Model is saved as a matrix rather than a Fixed/MixedEffect
            if idic[key].shape[1] == 1:
                idic[key] = FixedEffect(1) + FixedEffect(idic[key])
            else:
                idic[key] = (
                    FixedEffect(1)
                    + FixedEffect(idic[key][:, 0])
                    + MixedEffect(idic[key][:, 1])
                    + MixedEffect(1)
                )
        setattr(slm, key, idic[key])
        if key == "surf" and slm.surf is not None:
            slm.surf["tri"] += 1

    slm.fit(idic["Y"])

    # load expected outout data
    efile = open(expfile, "br")
    out = pickle.load(efile)
    efile.close()

    testout = []

    skip_keys = ["model", "correction", "_tri", "surf"]
    for key in out.keys():
        if key in skip_keys:
            continue
        if key == "P":
            testout.append(recursive_comparison(out[key], getattr(slm, key)))
        elif out[key] is not None:
            comp = np.allclose(out[key], getattr(slm, key), rtol=1e-05, equal_nan=True)
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 22
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("slm_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("slm_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
