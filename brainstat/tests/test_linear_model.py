"""Unit tests of linear_model."""
import pickle

import numpy as np
import pytest

from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect
from brainstat.tests.testutil import datadir


def dummy_test(infile, expfile):

    ifile = open(infile, "br")
    Din = pickle.load(ifile)
    ifile.close()

    Y = Din["Y"]
    M = Din["M"]

    # Convert M to a true BrainStat model
    fixed_effects = FixedEffect(M[:, Din["n_random"] :])
    if Din["n_random"] != 0:
        mixed_effects = MixedEffect(
            M[:, : Din["n_random"]],
            name_ran=["f" + str(x) for x in range(Din["n_random"])],
        )
        M = fixed_effects + mixed_effects
    else:
        M = fixed_effects

    # assign slm params
    slm = SLM(M, FixedEffect(1), surf=Din["surf"])

    # here we go --> run the linear model
    slm._linear_model(Y)

    ofile = open(expfile, "br")
    Dout = pickle.load(ofile)
    ofile.close()

    # compare...
    testout = []

    for k, v in Dout.items():
        if k == "surf":
            # Surface data is only stored for reconstruction in MATLAB.
            continue

        a = getattr(slm, k)

        comp = np.allclose(a, v, rtol=1e-05, equal_nan=True)
        testout.append(comp)
    assert all(testout)


expected_number_of_tests = 16
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xlinmod_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xlinmod_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
