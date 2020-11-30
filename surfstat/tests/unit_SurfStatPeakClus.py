import sys
sys.path.append("python")
from SurfStatPeakClus import *
import numpy as np
import sys
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    slm = {}
    slm['t']    = idic['t']
    slm['tri']  = idic['tri']

    mask = idic['mask']
    thresh = idic['thresh']
    reselspvert=None
    edg=None

    if 'reselspvert' in idic.keys():
        reselspvert = idic['reselspvert']

    if 'edg' in idic.keys():
        edg = idic['edg']

    if 'k' in idic.keys():
        slm['k'] = idic['k']

    if 'df' in idic.keys():
        slm['df'] = idic['df']


    # call python function
    P_peak, P_clus, P_clusid = py_SurfStatPeakClus(slm, mask, thresh,
                                                   reselspvert, edg)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    O_peak   = expdic['peak']
    O_clus   = expdic['clus']
    O_clusid = expdic['clusid']

    testout = []

    if isinstance(P_peak, (dict)):
        for key in P_peak.keys():
            comp = np.allclose(P_peak[key], O_peak[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)
    else:
        comp = np.allclose(P_peak, O_peak, rtol=1e-05, equal_nan=True)

    if isinstance(P_clus, (dict)):
        for key in P_clus.keys():
            comp = np.allclose(P_clus[key], O_clus[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(P_clus, O_clus, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    testout.append(np.allclose(P_clusid, O_clusid,
                               rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statpeakc_01_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statpeakc_02_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statpeakc_03_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statpeakc_04_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statpeakc_05_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statpeakc_06_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_06_OUT.pkl'
    dummy_test(infile, expfile)


def test_07():
    infile  = './tests/data/unitdata/statpeakc_07_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_07_OUT.pkl'
    dummy_test(infile, expfile)


def test_08():
    infile  = './tests/data/unitdata/statpeakc_08_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_08_OUT.pkl'
    dummy_test(infile, expfile)


def test_09():
    infile  = './tests/data/unitdata/statpeakc_09_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_09_OUT.pkl'
    dummy_test(infile, expfile)


def test_10():
    infile  = './tests/data/unitdata/statpeakc_10_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_10_OUT.pkl'
    dummy_test(infile, expfile)


def test_11():
    infile  = './tests/data/unitdata/statpeakc_11_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_11_OUT.pkl'
    dummy_test(infile, expfile)


def test_12():
    infile  = './tests/data/unitdata/statpeakc_12_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_12_OUT.pkl'
    dummy_test(infile, expfile)


def test_13():
    infile  = './tests/data/unitdata/statpeakc_13_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_13_OUT.pkl'
    dummy_test(infile, expfile)


def test_14():
    infile  = './tests/data/unitdata/statpeakc_14_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_14_OUT.pkl'
    dummy_test(infile, expfile)


def test_15():
    infile  = './tests/data/unitdata/statpeakc_15_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_15_OUT.pkl'
    dummy_test(infile, expfile)


def test_16():
    infile  = './tests/data/unitdata/statpeakc_16_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_16_OUT.pkl'
    dummy_test(infile, expfile)


def test_17():
    infile  = './tests/data/unitdata/statpeakc_17_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_17_OUT.pkl'
    dummy_test(infile, expfile)


def test_18():
    infile  = './tests/data/unitdata/statpeakc_18_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_18_OUT.pkl'
    dummy_test(infile, expfile)


def test_19():
    infile  = './tests/data/unitdata/statpeakc_19_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_19_OUT.pkl'
    dummy_test(infile, expfile)


def test_20():
    infile  = './tests/data/unitdata/statpeakc_20_IN.pkl'
    expfile = './tests/data/unitdata/statpeakc_20_OUT.pkl'
    dummy_test(infile, expfile)
