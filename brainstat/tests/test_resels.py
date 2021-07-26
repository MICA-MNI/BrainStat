"""Unit tests of compute_resels."""
import pickle

import numpy as np
import pytest

from brainstat.stats._multiple_comparisons import compute_resels
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from brainstat.tests.testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(FixedEffect(1), FixedEffect(1))
    for key in idic.keys():
        setattr(slm, key, idic[key])

    resels_py, reselspvert_py, edg_py = compute_resels(slm)

    out = {}
    out["resels"] = resels_py
    out["reselspvert"] = reselspvert_py
    out["edg"] = edg_py

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in out.keys():
        if out[key] is not None and expdic[key] is not None:
            comp = np.allclose(out[key], expdic[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 12
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xstatresl_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xstatresl_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
