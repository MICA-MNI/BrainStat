import sys
sys.path.append("python")
from SurfStatP import *
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
    slm['df']   = idic['df']
    slm['k']    = idic['k']
    slm['tri']  = idic['tri']

    mask = None
    clusthresh=0.001

    if 'dfs' in idic.keys():
        slm['dfs'] = idic['dfs']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'mask' in idic.keys():
        mask = idic['mask']

    if 'clusthresh' in idic.keys():
        clusthresh = idic['clusthresh']

    if 'X' in idic.keys():
        slm['X'] = idic['X']

    if 'coef' in idic.keys():
        slm['coef']  = idic['coef']

    if 'SSE' in idic.keys():
        slm['SSE'] = idic['SSE']

    if 'c' in idic.keys():
        slm['c'] = idic['c']

    if 'ef' in idic.keys():
        slm['ef'] = idic['ef']

    if 'sd' in idic.keys():
        slm['sd'] = idic['sd']


    PY_pval, PY_peak, PY_clus, PY_clusid = py_SurfStatP(slm, mask, clusthresh)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    O_pval   = expdic['pval']
    O_peak   = expdic['peak']
    O_clus   = expdic['clus']
    O_clusid = expdic['clusid']

    testout = []

    for key in PY_pval.keys():
        comp = np.allclose(PY_pval[key], O_pval[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    if isinstance(PY_peak, (dict)):
        for key in PY_peak.keys():
            comp = np.allclose(PY_peak[key], O_peak[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(PY_peak, O_peak, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    if isinstance(PY_peak, (dict)):
        for key in PY_clus.keys():
            comp = np.allclose(PY_clus[key], O_clus[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(PY_clus, O_clus, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    testout.append(np.allclose(PY_clusid, O_clusid,
                               rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statp_01_IN.pkl'
    expfile = './tests/data/unitdata/statp_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statp_02_IN.pkl'
    expfile = './tests/data/unitdata/statp_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statp_03_IN.pkl'
    expfile = './tests/data/unitdata/statp_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statp_04_IN.pkl'
    expfile = './tests/data/unitdata/statp_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statp_05_IN.pkl'
    expfile = './tests/data/unitdata/statp_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statp_06_IN.pkl'
    expfile = './tests/data/unitdata/statp_06_OUT.pkl'
    dummy_test(infile, expfile)


def test_07():
    infile  = './tests/data/unitdata/statp_07_IN.pkl'
    expfile = './tests/data/unitdata/statp_07_OUT.pkl'
    dummy_test(infile, expfile)


def test_08():
    infile  = './tests/data/unitdata/statp_08_IN.pkl'
    expfile = './tests/data/unitdata/statp_08_OUT.pkl'
    dummy_test(infile, expfile)


def test_09():
    infile  = './tests/data/unitdata/statp_09_IN.pkl'
    expfile = './tests/data/unitdata/statp_09_OUT.pkl'
    dummy_test(infile, expfile)


def test_10():
    infile  = './tests/data/unitdata/statp_10_IN.pkl'
    expfile = './tests/data/unitdata/statp_10_OUT.pkl'
    dummy_test(infile, expfile)


def test_11():
    infile  = './tests/data/unitdata/statp_11_IN.pkl'
    expfile = './tests/data/unitdata/statp_11_OUT.pkl'
    dummy_test(infile, expfile)


def test_12():
    infile  = './tests/data/unitdata/statp_12_IN.pkl'
    expfile = './tests/data/unitdata/statp_12_OUT.pkl'
    dummy_test(infile, expfile)


def test_13():
    infile  = './tests/data/unitdata/statp_13_IN.pkl'
    expfile = './tests/data/unitdata/statp_13_OUT.pkl'
    dummy_test(infile, expfile)


def test_14():
    infile  = './tests/data/unitdata/statp_14_IN.pkl'
    expfile = './tests/data/unitdata/statp_14_OUT.pkl'
    dummy_test(infile, expfile)


def test_15():
    infile  = './tests/data/unitdata/statp_15_IN.pkl'
    expfile = './tests/data/unitdata/statp_15_OUT.pkl'
    dummy_test(infile, expfile)


def test_16():
    infile  = './tests/data/unitdata/statp_16_IN.pkl'
    expfile = './tests/data/unitdata/statp_16_OUT.pkl'
    dummy_test(infile, expfile)


def test_17():
    infile  = './tests/data/unitdata/statp_17_IN.pkl'
    expfile = './tests/data/unitdata/statp_17_OUT.pkl'
    dummy_test(infile, expfile)


def test_18():
    infile  = './tests/data/unitdata/statp_18_IN.pkl'
    expfile = './tests/data/unitdata/statp_18_OUT.pkl'
    dummy_test(infile, expfile)


def test_19():
    infile  = './tests/data/unitdata/statp_19_IN.pkl'
    expfile = './tests/data/unitdata/statp_19_OUT.pkl'
    dummy_test(infile, expfile)


