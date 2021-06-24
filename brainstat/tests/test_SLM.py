import numpy as np
import pickle
import pytest
from brainstat.tests.testutil import datadir
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect


def recursive_dict_comparison(D1, D2):
    if len(D1.keys()) != len(D2.keys()):
        raise ValueError("Different number of keys in each dictionary.")

    output = True
    for key in D1.keys():
        if D1[key] is None and D2[key] is None:
            continue
        if isinstance(D1[key], dict):
            output = recursive_dict_comparison(D1[key], D2[key])
        elif isinstance(D1[key], list):
            output = np.all([np.all(x1 == x2) for x1, x2 in zip(D1[key], D2[key])])
        else:
            output = np.all(D1[key] == D2[key])
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
            testout.append(recursive_dict_comparison(out[key], getattr(slm, key)))
        elif out[key] is not None:
            comp = np.allclose(out[key], getattr(slm, key), rtol=1e-05, equal_nan=True)
            testout.append(comp)

    if not all(flag == True for (flag) in testout):
        [print(x,y) for x, y in zip(testout, out.keys())]

    assert all(flag == True for (flag) in testout)


expected_number_of_tests = 18
parametrize = pytest.mark.parametrize


@parametrize("test_number", range(1, expected_number_of_tests + 1))
def test_run_all(test_number):
    infile = datadir("slm_" + f"{test_number:02d}" + "_IN.pkl")
    expfile = datadir("slm_" + f"{test_number:02d}" + "_OUT.pkl")
    dummy_test(infile, expfile)
