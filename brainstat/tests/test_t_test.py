"""Unit tests of t-test."""
import pickle

import numpy as np
import pytest

from brainstat.stats.SLM import SLM
from brainstat.tests.testutil import array2effect, datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    model = array2effect(idic["M"], idic["n_random"])
    contrast = -idic["M"][:, -1]

    # run _t_test
    slm = SLM(model, contrast, idic["surf"])
    slm._linear_model(idic["Y"])
    slm._t_test()

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []
    for key in expdic.keys():
        if isinstance(expdic[key], dict):
            slm_sub_dict = getattr(slm, key)
            exp_sub_dict = expdic[key]
            comp = np.all(
                [
                    np.allclose(slm_sub_dict[x], exp_sub_dict[x])
                    for x in exp_sub_dict.keys()
                ]
            )
        else:
            comp = np.allclose(
                getattr(slm, key), expdic[key], rtol=1e-05, equal_nan=True
            )
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 16
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xstatt_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xstatt_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
