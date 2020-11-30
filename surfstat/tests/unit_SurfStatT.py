import sys
sys.path.append("python")
from SurfStatT import *
import numpy as np
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    slm = {}
    slm['X']    = idic['X']
    slm['df']   = idic['df']
    slm['coef'] = idic['coef']
    slm['SSE']  = idic['SSE']

    contrast = idic['contrast']

    if 'V' in idic.keys():
        slm['V']    = idic['V']

    if 'SSE' in idic.keys():
        slm['SSE']    = idic['SSE']

    if 'r' in idic.keys():
        slm['r']    = idic['r']

    if 'dr' in idic.keys():
        slm['dr']    = idic['dr']

    if 'tri' in idic.keys():
        slm['tri']    = idic['tri']

    if 'resl' in idic.keys():
        slm['resl']    = idic['resl']


    # run SurfStatT
    outdic = py_SurfStatT(slm, contrast)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statt_01_IN.pkl'
    expfile = './tests/data/unitdata/statt_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statt_02_IN.pkl'
    expfile = './tests/data/unitdata/statt_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statt_03_IN.pkl'
    expfile = './tests/data/unitdata/statt_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statt_04_IN.pkl'
    expfile = './tests/data/unitdata/statt_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statt_05_IN.pkl'
    expfile = './tests/data/unitdata/statt_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statt_06_IN.pkl'
    expfile = './tests/data/unitdata/statt_06_OUT.pkl'
    dummy_test(infile, expfile)


def test_07():
    infile  = './tests/data/unitdata/statt_07_IN.pkl'
    expfile = './tests/data/unitdata/statt_07_OUT.pkl'
    dummy_test(infile, expfile)


def test_08():
    infile  = './tests/data/unitdata/statt_08_IN.pkl'
    expfile = './tests/data/unitdata/statt_08_OUT.pkl'
    dummy_test(infile, expfile)


def test_09():
    infile  = './tests/data/unitdata/statt_09_IN.pkl'
    expfile = './tests/data/unitdata/statt_09_OUT.pkl'
    dummy_test(infile, expfile)


