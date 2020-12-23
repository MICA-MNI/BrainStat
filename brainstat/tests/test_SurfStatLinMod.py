import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatLinMod
from ..stats import Term
import gzip

def dummy_test(slm, oslm):

    testout = []

    for key in slm.keys():
        comp = np.allclose(slm[key], oslm[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    print(testout)

    assert all(flag == True for (flag) in testout)


def test_01():

    infile  = datadir('linmod_01_IN.pkl.gz')
    expfile = datadir('linmod_01_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_02():

    infile  = datadir('linmod_02_IN.pkl.gz')
    expfile = datadir('linmod_02_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_03():

    infile  = datadir('linmod_03_IN.pkl.gz')
    expfile = datadir('linmod_03_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_04():

    infile  = datadir('linmod_04_IN.pkl.gz')
    expfile = datadir('linmod_04_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_05():

    infile  = datadir('linmod_05_IN.pkl.gz')
    expfile = datadir('linmod_05_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_06():

    infile  = datadir('linmod_06_IN.pkl.gz')
    expfile = datadir('linmod_06_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_07():

    infile  = datadir('linmod_07_IN.pkl.gz')
    expfile = datadir('linmod_07_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = SurfStatLinMod(Y, M)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_08():

    infile  = datadir('linmod_08_IN.pkl.gz')
    expfile = datadir('linmod_08_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['tri'] = idic['tri']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_09():

    infile  = datadir('linmod_09_IN.pkl.gz')
    expfile = datadir('linmod_09_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['tri'] = idic['tri']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_10():

    infile  = datadir('linmod_10_IN.pkl.gz')
    expfile = datadir('linmod_10_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['lat'] = idic['lat']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_11():

    infile  = datadir('linmod_11_IN.pkl.gz')
    expfile = datadir('linmod_11_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['lat'] = idic['lat']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_12():

    infile  = datadir('linmod_12_IN.pkl.gz')
    expfile = datadir('linmod_12_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_13():

    infile  = datadir('linmod_13_IN.pkl.gz')
    expfile = datadir('linmod_13_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_14():

    infile  = datadir('linmod_14_IN.pkl.gz')
    expfile = datadir('linmod_14_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_15():

    infile  = datadir('linmod_15_IN.pkl.gz')
    expfile = datadir('linmod_15_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y        = idic['Y']
    params   = idic['params']
    colnames = list(idic['colnames'])
    M        = Term(params, colnames)

    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


def test_16():

    infile  = datadir('linmod_16_IN.pkl.gz')
    expfile = datadir('linmod_16_OUT.pkl.gz')

    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y        = idic['Y']
    params   = idic['params']
    M        = Term(params)

    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    with gzip.open(expfile, 'rb') as f:
        oslm  = pickle.load(f)

    dummy_test(slm, oslm)


