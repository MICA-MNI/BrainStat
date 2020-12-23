import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatPeakClus
import gzip


def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic = pickle.load(f)

    slm = {}
    slm['t'] = idic['t']
    slm['tri'] = idic['tri']

    mask = idic['mask']
    thresh = idic['thresh']
    reselspvert = None
    edg = None

    if 'reselspvert' in idic.keys():
        reselspvert = idic['reselspvert']

    if 'edg' in idic.keys():
        edg = idic['edg']

    if 'k' in idic.keys():
        slm['k'] = idic['k']

    if 'df' in idic.keys():
        slm['df'] = idic['df']

    # call python function
    P_peak, P_clus, P_clusid = SurfStatPeakClus(slm, mask, thresh,
                                                reselspvert, edg)

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic = pickle.load(f)

    O_peak = expdic['peak']
    O_clus = expdic['clus']
    O_clusid = expdic['clusid']

    testout = []

    if isinstance(P_peak, (dict)):
        for key in P_peak.keys():
            comp = np.allclose(P_peak[key], O_peak[key],
                               rtol=1e-05, equal_nan=True)
            testout.append(comp)
    else:
        comp = np.allclose(P_peak, O_peak, rtol=1e-05, equal_nan=True)

    if isinstance(P_clus, (dict)):
        for key in P_clus.keys():
            comp = np.allclose(P_clus[key], O_clus[key],
                               rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(P_clus, O_clus, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    testout.append(np.allclose(P_clusid, O_clusid,
                               rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    infile = datadir('statpeakc_01_IN.pkl.gz')
    expfile = datadir('statpeakc_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile = datadir('statpeakc_02_IN.pkl.gz')
    expfile = datadir('statpeakc_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile = datadir('statpeakc_03_IN.pkl.gz')
    expfile = datadir('statpeakc_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile = datadir('statpeakc_04_IN.pkl.gz')
    expfile = datadir('statpeakc_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_05():
    infile = datadir('statpeakc_05_IN.pkl.gz')
    expfile = datadir('statpeakc_05_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_06():
    infile = datadir('statpeakc_06_IN.pkl.gz')
    expfile = datadir('statpeakc_06_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_07():
    infile = datadir('statpeakc_07_IN.pkl.gz')
    expfile = datadir('statpeakc_07_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_08():
    infile = datadir('statpeakc_08_IN.pkl.gz')
    expfile = datadir('statpeakc_08_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_09():
    infile = datadir('statpeakc_09_IN.pkl.gz')
    expfile = datadir('statpeakc_09_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_10():
    infile = datadir('statpeakc_10_IN.pkl.gz')
    expfile = datadir('statpeakc_10_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_11():
    infile = datadir('statpeakc_11_IN.pkl.gz')
    expfile = datadir('statpeakc_11_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_12():
    infile = datadir('statpeakc_12_IN.pkl.gz')
    expfile = datadir('statpeakc_12_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_13():
    infile = datadir('statpeakc_13_IN.pkl.gz')
    expfile = datadir('statpeakc_13_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_14():
    infile = datadir('statpeakc_14_IN.pkl.gz')
    expfile = datadir('statpeakc_14_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_15():
    infile = datadir('statpeakc_15_IN.pkl.gz')
    expfile = datadir('statpeakc_15_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_16():
    infile = datadir('statpeakc_16_IN.pkl.gz')
    expfile = datadir('statpeakc_16_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_17():
    infile = datadir('statpeakc_17_IN.pkl.gz')
    expfile = datadir('statpeakc_17_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_18():
    infile = datadir('statpeakc_18_IN.pkl.gz')
    expfile = datadir('statpeakc_18_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_19():
    infile = datadir('statpeakc_19_IN.pkl.gz')
    expfile = datadir('statpeakc_19_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_20():
    infile = datadir('statpeakc_20_IN.pkl.gz')
    expfile = datadir('statpeakc_20_OUT.pkl.gz')
    dummy_test(infile, expfile)
