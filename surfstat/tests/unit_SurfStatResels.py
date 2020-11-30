import sys
sys.path.append("python")
from SurfStatResels import *
import numpy as np
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    slm = {}

    if 'tri' in idic.keys():
        slm['tri'] = idic['tri']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'lat' in idic.keys():
        slm['lat'] = idic['lat']



    mask = None

    if 'mask' in idic.keys():
        mask = idic['mask']

    resels_py, reselspvert_py, edg_py =  py_SurfStatResels(slm,mask)

    out = {}
    out['resels']      = resels_py
    out['reselspvert'] = reselspvert_py
    out['edg']         = edg_py

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in out.keys():
        if out[key] is not None and expdic[key] is not None:
            comp = np.allclose(out[key], expdic[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statresl_01_IN.pkl'
    expfile = './tests/data/unitdata/statresl_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statresl_02_IN.pkl'
    expfile = './tests/data/unitdata/statresl_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statresl_03_IN.pkl'
    expfile = './tests/data/unitdata/statresl_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statresl_04_IN.pkl'
    expfile = './tests/data/unitdata/statresl_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statresl_05_IN.pkl'
    expfile = './tests/data/unitdata/statresl_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statresl_06_IN.pkl'
    expfile = './tests/data/unitdata/statresl_06_OUT.pkl'
    dummy_test(infile, expfile)


def test_07():
    infile  = './tests/data/unitdata/statresl_07_IN.pkl'
    expfile = './tests/data/unitdata/statresl_07_OUT.pkl'
    dummy_test(infile, expfile)


def test_08():
    infile  = './tests/data/unitdata/statresl_08_IN.pkl'
    expfile = './tests/data/unitdata/statresl_08_OUT.pkl'
    dummy_test(infile, expfile)


def test_09():
    infile  = './tests/data/unitdata/statresl_09_IN.pkl'
    expfile = './tests/data/unitdata/statresl_09_OUT.pkl'
    dummy_test(infile, expfile)


def test_10():
    infile  = './tests/data/unitdata/statresl_10_IN.pkl'
    expfile = './tests/data/unitdata/statresl_10_OUT.pkl'
    dummy_test(infile, expfile)


def test_11():
    infile  = './tests/data/unitdata/statresl_11_IN.pkl'
    expfile = './tests/data/unitdata/statresl_11_OUT.pkl'
    dummy_test(infile, expfile)


def test_12():
    infile  = './tests/data/unitdata/statresl_12_IN.pkl'
    expfile = './tests/data/unitdata/statresl_12_OUT.pkl'
    dummy_test(infile, expfile)


def test_13():
    infile  = './tests/data/unitdata/statresl_13_IN.pkl'
    expfile = './tests/data/unitdata/statresl_13_OUT.pkl'
    dummy_test(infile, expfile)




