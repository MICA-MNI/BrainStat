"""Unit tests of fdr."""

import pickle

import numpy as np
import pytest

from brainstat.stats._multiple_comparisons import _fdr
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

    # run fdr
    Q = _fdr(slm)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    assert np.allclose(Q, expdic["Q"])


expected_number_of_tests = 14
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xstatq_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xstatq_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
