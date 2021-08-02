"""Unit tests of random_field_theory."""
import pickle

import numpy as np
import pytest

from brainstat.stats._multiple_comparisons import _random_field_theory
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
        if key == "clusthresh":
            slm.cluster_threshold = idic[key]
        else:
            setattr(slm, key, idic[key])

    empirical_output = _random_field_theory(slm)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()
    expected_output = (expdic["pval"], expdic["peak"], expdic["clus"], expdic["clusid"])

    testout = []
    for (empirical, expected) in zip(empirical_output, expected_output):
        if isinstance(expected, dict):
            for key in expected:
                if key == "mask":
                    continue
                if empirical[key] is None:
                    testout.append(expected[key] is None)
                else:
                    comp = np.allclose(
                        empirical[key], expected[key], rtol=1e-05, equal_nan=True
                    )
                    testout.append(comp)
        else:
            if empirical is None:
                testout.append(expected is None)
            else:
                if len(expected) != 0:
                    comp = np.allclose(empirical, expected, rtol=1e-05, equal_nan=True)
                    testout.append(comp)

    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 21
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xstatp_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xstatp_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
