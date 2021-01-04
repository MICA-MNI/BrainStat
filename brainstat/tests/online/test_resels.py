import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.models import _resels


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

    resels_py, reselspvert_py, edg_py =  _resels(slm,mask)

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
    # ['tri'] :np array, shape (4, 3), int64
    infile  = datadir('statresl_01_IN.pkl')
    expfile = datadir('statresl_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    # ['tri'] :np array, shape (4, 3), int64
    # ['resl'] :np array, shape (8, 6), float64
    infile  = datadir('statresl_02_IN.pkl')
    expfile = datadir('statresl_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    # ['tri'] :np array, shape (4, 3), int64
    # ['resl'] :np array, shape (8, 6), float64
    # ['mask'] :np array, shape (5,), bool
    infile  = datadir('statresl_03_IN.pkl')
    expfile = datadir('statresl_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    # ['lat'] :np array, shape (10, 10, 10), float64
    infile  = datadir('statresl_04_IN.pkl')
    expfile = datadir('statresl_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05():
    # ['lat'] :np array, shape (10, 10, 10), bool
    infile  = datadir('statresl_05_IN.pkl')
    expfile = datadir('statresl_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['mask'] :np array, shape (457,), bool
    infile  = datadir('statresl_06_IN.pkl')
    expfile = datadir('statresl_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1359, 1), float64
    infile  = datadir('statresl_07_IN.pkl')
    expfile = datadir('statresl_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1251, 1), float64
    # ['mask'] :np array, shape (499,), bool
    infile  = datadir('statresl_08_IN.pkl')
    expfile = datadir('statresl_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1198, 1), float64
    # ['mask'] :np array, shape (478,), bool
    infile  = datadir('statresl_09_IN.pkl')
    expfile = datadir('statresl_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10():
    # ['tri'] :np array, shape (129960, 3), int32
    # ['resl'] :np array, shape (194940, 1), float64
    infile  = datadir('statresl_10_IN.pkl')
    expfile = datadir('statresl_10_OUT.pkl')
    dummy_test(infile, expfile)


def test_11():
    # ['tri'] :np array, shape (129960, 3), int32
    # ['resl'] :np array, shape (194940, 1), float64
    # ['mask'] :np array, shape (64984,), bool
    infile  = datadir('statresl_11_IN.pkl')
    expfile = datadir('statresl_11_OUT.pkl')
    dummy_test(infile, expfile)


def test_12():
    # ['tri'] :np array, shape (129960, 3), int32
    # ['resl'] :np array, shape (194940, 1), float64
    infile  = datadir('statresl_12_IN.pkl')
    expfile = datadir('statresl_12_OUT.pkl')
    dummy_test(infile, expfile)


def test_13():
    # ['tri'] :np array, shape (129960, 3), int32
    # ['resl'] :np array, shape (194940, 1), float64
    # ['mask'] :np array, shape (64984,), bool
    infile  = datadir('statresl_13_IN.pkl')
    expfile = datadir('statresl_13_OUT.pkl')
    dummy_test(infile, expfile)


