import sys
sys.path.append("python")
from SurfStatF import *
import numpy as np
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    slm1 = {}
    slm1['X']    = idic['slm1X']
    slm1['df']   = idic['slm1df']
    slm1['SSE']  = idic['slm1SSE']
    slm1['coef'] = idic['slm1coef']

    slm2 = {}
    slm2['X']    = idic['slm2X']
    slm2['df']   = idic['slm2df']
    slm2['SSE']  = idic['slm2SSE']
    slm2['coef'] = idic['slm2coef']

    # check other potential keys in input
    if 'slm1tri' in idic.keys():
        slm1['tri']    = idic['slm1tri']
        slm2['tri']    = idic['slm2tri']

    if 'slm1resl' in idic.keys():
        slm1['resl'] = idic['slm1resl']
        slm2['resl'] = idic['slm2resl']

    if 'slm1c' in idic.keys():
        slm1['c']    = idic['slm1c']
        slm2['c']    = idic['slm2c']

    if 'slm1k' in idic.keys():
        slm1['k']    = idic['slm1k']
        slm2['k']    = idic['slm2k']

    if 'slm1ef' in idic.keys():
        slm1['ef']    = idic['slm1ef']
        slm2['ef']    = idic['slm2ef']

    if 'slm1sd' in idic.keys():
        slm1['sd']    = idic['slm1sd']
        slm2['sd']    = idic['slm2sd']

    if 'slm1t' in idic.keys():
        slm1['t']    = idic['slm1t']
        slm2['t']    = idic['slm2t']



    # run SurfStatF
    outdic = py_SurfStatF(slm1, slm2)

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
    infile  = './tests/data/unitdata/statf_01_IN.pkl'
    expfile = './tests/data/unitdata/statf_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statf_02_IN.pkl'
    expfile = './tests/data/unitdata/statf_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statf_03_IN.pkl'
    expfile = './tests/data/unitdata/statf_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statf_04_IN.pkl'
    expfile = './tests/data/unitdata/statf_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statf_05_IN.pkl'
    expfile = './tests/data/unitdata/statf_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statf_06_IN.pkl'
    expfile = './tests/data/unitdata/statf_06_OUT.pkl'
    dummy_test(infile, expfile)


def test_07():
    infile  = './tests/data/unitdata/statf_07_IN.pkl'
    expfile = './tests/data/unitdata/statf_07_OUT.pkl'
    dummy_test(infile, expfile)


def test_08():
    infile  = './tests/data/unitdata/statf_08_IN.pkl'
    expfile = './tests/data/unitdata/statf_08_OUT.pkl'
    dummy_test(infile, expfile)


def test_09():
    infile  = './tests/data/unitdata/statf_09_IN.pkl'
    expfile = './tests/data/unitdata/statf_09_OUT.pkl'
    dummy_test(infile, expfile)


def test_10():
    infile  = './tests/data/unitdata/statf_10_IN.pkl'
    expfile = './tests/data/unitdata/statf_10_OUT.pkl'
    dummy_test(infile, expfile)


def test_11():
    infile  = './tests/data/unitdata/statf_11_IN.pkl'
    expfile = './tests/data/unitdata/statf_11_OUT.pkl'
    dummy_test(infile, expfile)


def test_12():
    infile  = './tests/data/unitdata/statf_12_IN.pkl'
    expfile = './tests/data/unitdata/statf_12_OUT.pkl'
    dummy_test(infile, expfile)



