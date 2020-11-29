import sys
sys.path.append("python")
import surfstat_wrap as sw
from SurfStatQ import *
import numpy as np
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    slm = {}
    slm['t']  = idic['t']
    slm['df'] = idic['df']
    slm['k']  = idic['k']

    # check other potential keys in input
    if 'tri' in idic.keys():
        slm['tri']    = idic['tri']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'dfs' in idic.keys():
        slm['dfs'] = idic['dfs']

    if 'du' in idic.keys():
        slm['du'] = idic['du']

    if 'c' in idic.keys():
        slm['c']    = idic['c']

    if 'k' in idic.keys():
        slm['k']    = idic['k']

    if 'ef' in idic.keys():
        slm['ef']    = idic['ef']

    if 'sd' in idic.keys():
        slm['sd']    = idic['sd']

    if 't' in idic.keys():
        slm['t']    = idic['t']

    if 'SSE' in idic.keys():
        slm['SSE']    = idic['SSE']

    if 'mask' in idic.keys():
        mask = idic['mask']
    else:
        mask = None


    # run SurfStatF
    outdic = py_SurfStatQ(slm, mask)

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
    infile  = './tests/data/unitdata/statq_01_IN.pkl'
    expfile = './tests/data/unitdata/statq_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statq_02_IN.pkl'
    expfile = './tests/data/unitdata/statq_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statq_03_IN.pkl'
    expfile = './tests/data/unitdata/statq_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statq_04_IN.pkl'
    expfile = './tests/data/unitdata/statq_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statq_05_IN.pkl'
    expfile = './tests/data/unitdata/statq_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statq_06_IN.pkl'
    expfile = './tests/data/unitdata/statq_06_OUT.pkl'
    dummy_test(infile, expfile)


def test_07():
    infile  = './tests/data/unitdata/statq_07_IN.pkl'
    expfile = './tests/data/unitdata/statq_07_OUT.pkl'
    dummy_test(infile, expfile)


def test_08():
    infile  = './tests/data/unitdata/statq_08_IN.pkl'
    expfile = './tests/data/unitdata/statq_08_OUT.pkl'
    dummy_test(infile, expfile)


def test_09():
    infile  = './tests/data/unitdata/statq_09_IN.pkl'
    expfile = './tests/data/unitdata/statq_09_OUT.pkl'
    dummy_test(infile, expfile)


def test_10():
    infile  = './tests/data/unitdata/statq_10_IN.pkl'
    expfile = './tests/data/unitdata/statq_10_OUT.pkl'
    dummy_test(infile, expfile)


def test_11():
    infile  = './tests/data/unitdata/statq_11_IN.pkl'
    expfile = './tests/data/unitdata/statq_11_OUT.pkl'
    dummy_test(infile, expfile)


def test_12():
    infile  = './tests/data/unitdata/statq_12_IN.pkl'
    expfile = './tests/data/unitdata/statq_12_OUT.pkl'
    dummy_test(infile, expfile)


def test_13():
    infile  = './tests/data/unitdata/statq_13_IN.pkl'
    expfile = './tests/data/unitdata/statq_13_OUT.pkl'
    dummy_test(infile, expfile)


def test_14():
    infile  = './tests/data/unitdata/statq_14_IN.pkl'
    expfile = './tests/data/unitdata/statq_14_OUT.pkl'
    dummy_test(infile, expfile)



