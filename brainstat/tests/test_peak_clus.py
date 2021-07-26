"""Unit tests of peak_clus."""
import pickle

import numpy as np
import pytest

from brainstat.stats._multiple_comparisons import peak_clus
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(FixedEffect(1), FixedEffect(1))
    slm.t = idic["t"]
    slm.tri = idic["tri"]
    slm.mask = idic["mask"]
    slm.df = idic["df"]
    slm.k = idic["k"]
    slm.resl = idic["resl"]

    thresh = idic["thresh"]
    reselspvert = idic["reselspvert"]
    edg = idic["edg"]

    # call python function
    P_peak, P_clus, P_clusid = peak_clus(slm, thresh, reselspvert, edg)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    O_peak = expdic["peak"]
    O_clus = expdic["clus"]
    O_clusid = expdic["clusid"]

    testout = []

    if isinstance(P_peak, dict):
        for key in P_peak.keys():
            comp = np.allclose(P_peak[key], O_peak[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)
    else:
        comp = np.allclose(P_peak, O_peak, rtol=1e-05, equal_nan=True)

    if isinstance(P_clus, dict):
        for key in P_clus.keys():
            comp = np.allclose(P_clus[key], O_clus[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(P_clus, O_clus, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    testout.append(np.allclose(P_clusid, O_clusid, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 72
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xstatpeakc_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xstatpeakc_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
