import testutil
import sys
sys.path.append("brainstat/stats")
from SurfStatQ import SurfStatQ
import numpy as np
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


    # run SurfStatQ
    outdic = SurfStatQ(slm, mask)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)

datadir = testutil.datadir


def test_01(datadir):
    infile  = datadir.join('statq_01_IN.pkl')
    expfile = datadir.join('statq_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02(datadir):
    infile  = datadir.join('statq_02_IN.pkl')
    expfile = datadir.join('statq_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03(datadir):
    infile  = datadir.join('statq_03_IN.pkl')
    expfile = datadir.join('statq_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04(datadir):
    infile  = datadir.join('statq_04_IN.pkl')
    expfile = datadir.join('statq_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05(datadir):
    infile  = datadir.join('statq_05_IN.pkl')
    expfile = datadir.join('statq_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06(datadir):
    infile  = datadir.join('statq_06_IN.pkl')
    expfile = datadir.join('statq_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07(datadir):
    infile  = datadir.join('statq_07_IN.pkl')
    expfile = datadir.join('statq_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08(datadir):
    infile  = datadir.join('statq_08_IN.pkl')
    expfile = datadir.join('statq_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09(datadir):
    infile  = datadir.join('statq_09_IN.pkl')
    expfile = datadir.join('statq_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10(datadir):
    infile  = datadir.join('statq_10_IN.pkl')
    expfile = datadir.join('statq_10_OUT.pkl')
    dummy_test(infile, expfile)


def test_11(datadir):
    infile  = datadir.join('statq_11_IN.pkl')
    expfile = datadir.join('statq_11_OUT.pkl')
    dummy_test(infile, expfile)


def test_12(datadir):
    infile  = datadir.join('statq_12_IN.pkl')
    expfile = datadir.join('statq_12_OUT.pkl')
    dummy_test(infile, expfile)


def test_13(datadir):
    infile  = datadir.join('statq_13_IN.pkl')
    expfile = datadir.join('statq_13_OUT.pkl')
    dummy_test(infile, expfile)


def test_14(datadir):
    infile  = datadir.join('statq_14_IN.pkl')
    expfile = datadir.join('statq_14_OUT.pkl')
    dummy_test(infile, expfile)









