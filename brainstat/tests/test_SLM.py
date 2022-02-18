"""Unit tests of SLM."""
import pickle
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import templateflow.api as tflow

from brainstat.stats.SLM import SLM, _onetailed_to_twotailed
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
    elif isinstance(X1, pd.DataFrame):
        iterator = zip(X1.values.tolist(), X2.values.tolist())
    else:
        # Assume not iterable.
        iterator = zip([X1], [X2])

    output = True
    for x, y in iterator:
        if x is None and y is None:
            output = True
        elif isinstance(x, list) or isinstance(x, dict) or isinstance(x, pd.DataFrame):
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

    # Format of self.P changed since files were created -- alter out to match some changes.
    # Combine the list outputs, sort with pandas, and return to list.
    if "P" in out:
        out["P"]["pval"]["C"] = _onetailed_to_twotailed(
            out["P"]["pval"]["C"][0], out["P"]["pval"]["C"][1]
        )

        for key1 in ["peak", "clus"]:
            P_tmp = []
            none_squeeze = lambda x: np.squeeze(x) if x is not None else None
            for i in range(len(out["P"][key1]["P"])):
                tail_dict = {
                    key: none_squeeze(value[i]) for key, value in out["P"][key1].items()
                }
                if tail_dict["P"] is not None:
                    if tail_dict["P"].size == 1:
                        P_tmp.append(pd.DataFrame.from_dict([tail_dict]))
                    else:
                        P_tmp.append(pd.DataFrame.from_dict(tail_dict))
                        P_tmp[i].sort_values(by="P", ascending=True)
                else:
                    P_tmp.append(pd.DataFrame(columns=tail_dict.keys()))
            out["P"][key1] = P_tmp

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


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Template flow has issues on windows."
)
def test_volumetric_input():
    mask_image = nib.load(
        tflow.get("MNI152Lin", resolution="02", desc="brain", suffix="mask")
    )
    n_voxels = (mask_image.get_fdata() != 0).sum()
    n_subjects = 3
    data = np.random.rand(n_subjects, n_voxels)
    model = FixedEffect(1)
    contrast = np.ones(3)

    slm = SLM(model, contrast, surf=mask_image)
    slm.fit(data)
