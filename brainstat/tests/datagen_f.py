"""Data generation for f-test unit tests."""

import pickle

import numpy as np
from sklearn.model_selection import ParameterGrid

from brainstat.stats.SLM import SLM, f_test
from brainstat.stats.terms import FixedEffect
from brainstat.tests.testutil import datadir


def generate_random_two_slms(I):
    slm1 = SLM(FixedEffect(1), FixedEffect(1))
    slm2 = SLM(FixedEffect(1), FixedEffect(2))

    for key in I.keys():
        if "1" in key:
            setattr(slm1, key[4:], I[key])
        elif "2" in key:
            setattr(slm2, key[4:], I[key])
    return slm1, slm2


def generate_f_test_out(slm1, slm2):
    D = {}
    slm_OUT = f_test(slm1, slm2)
    mykeys = ["X", "df", "SSE", "coef", "k", "t"]
    for key in mykeys:
        D[key] = getattr(slm_OUT, key)
    return D


def params2files(I, D, test_num):
    """Converts params to input/output files"""
    basename = "xstatf"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)
    return


def generate_test_data():
    np.random.seed(0)

    a = 40
    b = 30
    c = 100
    d = 42
    e = 77

    mygrid = [
        {
            "slm1X": [np.random.randint(0, a, size=(a, b)), np.random.rand(a, b)],
            "slm1df": [int(a)],
            "slm2df": [int(b)],
            "slm1SSE": [
                np.random.randint(1, a, size=(c, 1)),
                np.random.randint(1, a, size=(c, d)),
            ],
            "slm2SSE": [
                np.random.randint(1, a, size=(c, 1)),
                np.random.randint(1, a, size=(c, d)),
            ],
            "slm1coef": [np.random.rand(e, d), np.random.rand(e, d, 3)],
            "slm2coef": [np.random.rand(e, d)],
        }
    ]

    myparamgrid = ParameterGrid(mygrid)

    # Here wo go!
    # Test 1-16 : slm1X 2D array type int or float64, slm1df and slm2df int,
    # slm1SSE and slm2SSE 2D arrays type int, slm1coef and slm2coef 2D or 3D
    # arrays type float64

    test_num = 0
    for params in myparamgrid:
        test_num += 1
        I = {}
        for key in params.keys():
            I[key] = params[key]
        # make "slm2X" equal to "slm1X" to avoid nested-model error
        I["slm2X"] = I["slm1X"]
        slm1, slm2 = generate_random_two_slms(I)
        D = generate_f_test_out(slm1, slm2)
        params2files(I, D, test_num)


if __name__ == "__main__":
    generate_test_data()
