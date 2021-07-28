"""Unit tests of stat_threshold."""
import pickle

import numpy as np

from brainstat.stats._multiple_comparisons import stat_threshold

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    # run stat_xthreshold
    A, B, C, D, E, F = stat_threshold(
        idic["search_volume"],
        idic["num_voxels"],
        idic["fwhm"],
        idic["df"],
        idic["p_val_peak"],
        idic["cluster_threshold"],
        idic["p_val_extent"],
        idic["nconj"],
        idic["nvar"],
        None,
        idic["nprint"],
    )
    outdic = {
        "peak_threshold": A,
        "extent_threshold": B,
        "peak_threshold_1": C,
        "extent_threshold_1": D,
        "t": E,
        "rho": F,
    }

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


# parameters in *pck is equal to default params, if not specified in tests


def test_01():
    infile = datadir("xthresh_01_IN.pkl")
    expfile = datadir("xthresh_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xthresh_02_IN.pkl")
    expfile = datadir("xthresh_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xthresh_03_IN.pkl")
    expfile = datadir("xthresh_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xthresh_04_IN.pkl")
    expfile = datadir("xthresh_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xthresh_05_IN.pkl")
    expfile = datadir("xthresh_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    infile = datadir("xthresh_06_IN.pkl")
    expfile = datadir("xthresh_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    infile = datadir("xthresh_07_IN.pkl")
    expfile = datadir("xthresh_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    infile = datadir("xthresh_08_IN.pkl")
    expfile = datadir("xthresh_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    infile = datadir("xthresh_09_IN.pkl")
    expfile = datadir("xthresh_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    infile = datadir("xthresh_10_IN.pkl")
    expfile = datadir("xthresh_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    infile = datadir("xthresh_11_IN.pkl")
    expfile = datadir("xthresh_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    infile = datadir("xthresh_12_IN.pkl")
    expfile = datadir("xthresh_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    infile = datadir("xthresh_13_IN.pkl")
    expfile = datadir("xthresh_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    infile = datadir("xthresh_14_IN.pkl")
    expfile = datadir("xthresh_14_OUT.pkl")
    dummy_test(infile, expfile)


def test_15():
    infile = datadir("xthresh_15_IN.pkl")
    expfile = datadir("xthresh_15_OUT.pkl")
    dummy_test(infile, expfile)


def test_16():
    infile = datadir("xthresh_16_IN.pkl")
    expfile = datadir("xthresh_16_OUT.pkl")
    dummy_test(infile, expfile)


def test_17():
    infile = datadir("xthresh_17_IN.pkl")
    expfile = datadir("xthresh_17_OUT.pkl")
    dummy_test(infile, expfile)


def test_18():
    infile = datadir("xthresh_18_IN.pkl")
    expfile = datadir("xthresh_18_OUT.pkl")
    dummy_test(infile, expfile)


def test_19():
    infile = datadir("xthresh_19_IN.pkl")
    expfile = datadir("xthresh_19_OUT.pkl")
    dummy_test(infile, expfile)


def test_20():
    infile = datadir("xthresh_20_IN.pkl")
    expfile = datadir("xthresh_20_OUT.pkl")
    dummy_test(infile, expfile)


def test_21():
    infile = datadir("xthresh_21_IN.pkl")
    expfile = datadir("xthresh_21_OUT.pkl")
    dummy_test(infile, expfile)


def test_22():
    infile = datadir("xthresh_22_IN.pkl")
    expfile = datadir("xthresh_22_OUT.pkl")
    dummy_test(infile, expfile)


def test_23():
    infile = datadir("xthresh_23_IN.pkl")
    expfile = datadir("xthresh_23_OUT.pkl")
    dummy_test(infile, expfile)


def test_24():
    infile = datadir("xthresh_24_IN.pkl")
    expfile = datadir("xthresh_24_OUT.pkl")
    dummy_test(infile, expfile)


def test_25():
    infile = datadir("xthresh_25_IN.pkl")
    expfile = datadir("xthresh_25_OUT.pkl")
    dummy_test(infile, expfile)


def test_26():
    infile = datadir("xthresh_26_IN.pkl")
    expfile = datadir("xthresh_26_OUT.pkl")
    dummy_test(infile, expfile)


def test_27():
    infile = datadir("xthresh_27_IN.pkl")
    expfile = datadir("xthresh_27_OUT.pkl")
    dummy_test(infile, expfile)


def test_28():
    infile = datadir("xthresh_28_IN.pkl")
    expfile = datadir("xthresh_28_OUT.pkl")
    dummy_test(infile, expfile)


def test_29():
    infile = datadir("xthresh_29_IN.pkl")
    expfile = datadir("xthresh_29_OUT.pkl")
    dummy_test(infile, expfile)


def test_30():
    infile = datadir("xthresh_30_IN.pkl")
    expfile = datadir("xthresh_30_OUT.pkl")
    dummy_test(infile, expfile)


def test_31():
    infile = datadir("xthresh_31_IN.pkl")
    expfile = datadir("xthresh_31_OUT.pkl")
    dummy_test(infile, expfile)


def test_32():
    infile = datadir("xthresh_32_IN.pkl")
    expfile = datadir("xthresh_32_OUT.pkl")
    dummy_test(infile, expfile)


def test_33():
    infile = datadir("xthresh_33_IN.pkl")
    expfile = datadir("xthresh_33_OUT.pkl")
    dummy_test(infile, expfile)


def test_34():
    infile = datadir("xthresh_34_IN.pkl")
    expfile = datadir("xthresh_34_OUT.pkl")
    dummy_test(infile, expfile)


def test_35():
    infile = datadir("xthresh_35_IN.pkl")
    expfile = datadir("xthresh_35_OUT.pkl")
    dummy_test(infile, expfile)


def test_36():
    infile = datadir("xthresh_36_IN.pkl")
    expfile = datadir("xthresh_36_OUT.pkl")
    dummy_test(infile, expfile)


def test_37():
    infile = datadir("xthresh_37_IN.pkl")
    expfile = datadir("xthresh_37_OUT.pkl")
    dummy_test(infile, expfile)


def test_38():
    infile = datadir("xthresh_38_IN.pkl")
    expfile = datadir("xthresh_38_OUT.pkl")
    dummy_test(infile, expfile)


def test_39():
    infile = datadir("xthresh_39_IN.pkl")
    expfile = datadir("xthresh_39_OUT.pkl")
    dummy_test(infile, expfile)


def test_40():
    infile = datadir("xthresh_40_IN.pkl")
    expfile = datadir("xthresh_40_OUT.pkl")
    dummy_test(infile, expfile)


def test_41():
    infile = datadir("xthresh_41_IN.pkl")
    expfile = datadir("xthresh_41_OUT.pkl")
    dummy_test(infile, expfile)


def test_42():
    infile = datadir("xthresh_42_IN.pkl")
    expfile = datadir("xthresh_42_OUT.pkl")
    dummy_test(infile, expfile)


def test_43():
    infile = datadir("xthresh_43_IN.pkl")
    expfile = datadir("xthresh_43_OUT.pkl")
    dummy_test(infile, expfile)


def test_44():
    infile = datadir("xthresh_44_IN.pkl")
    expfile = datadir("xthresh_44_OUT.pkl")
    dummy_test(infile, expfile)


def test_45():
    infile = datadir("xthresh_45_IN.pkl")
    expfile = datadir("xthresh_45_OUT.pkl")
    dummy_test(infile, expfile)


def test_46():
    infile = datadir("xthresh_46_IN.pkl")
    expfile = datadir("xthresh_46_OUT.pkl")
    dummy_test(infile, expfile)


def test_47():
    infile = datadir("xthresh_47_IN.pkl")
    expfile = datadir("xthresh_47_OUT.pkl")
    dummy_test(infile, expfile)


def test_48():
    infile = datadir("xthresh_48_IN.pkl")
    expfile = datadir("xthresh_48_OUT.pkl")
    dummy_test(infile, expfile)
