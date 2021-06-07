import numpy as np
import pytest
from brainstat.tests.testutil import datadir
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM
import pickle


def dummy_test(infile, expfile, simple=True):

    ifile = open(infile, "br")
    Din = pickle.load(ifile)
    ifile.close()

    Y = Din["Y"]
    M = Din["M"]

    # assign slm params
    slm = SLM(M, FixedEffect(1))

    if "tri" in Din:
        slm.surf = {"tri": Din["tri"]}
    if "lat" in Din:
        slm.surf = {"lat": Din["lat"]}

    # here we go --> run the linear model
    slm.linear_model(Y)

    ofile = open(expfile, "br")
    Dout = pickle.load(ofile)
    ofile.close()

    # compare...
    testout = []
    for makey_ in Dout.keys():
        comp = np.allclose(
            getattr(slm, makey_), Dout[makey_], rtol=1e-05, equal_nan=True
        )
        testout.append(comp)
    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 15
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("xlinmod_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("xlinmod_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
