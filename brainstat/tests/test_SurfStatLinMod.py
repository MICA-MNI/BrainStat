import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatLinMod
from ..stats import Term


def dummy_test(slm, oslm):

    testout = []

    for key in slm.keys():
        comp = np.allclose(slm[key], oslm[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    print(testout)

    assert all(flag == True for (flag) in testout)



def test_01():
    # ['Y'] : numpy array, shape (43, 43), dtype('float64')
    # ['M'] : numpy array, (43, 43), dtype('float64')
    infile  = datadir('linmod_01_IN.pkl')
    expfile = datadir('linmod_01_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_02():
    # ['Y'] : numpy array, (62, 7), dtype('float64')
    # ['M'] : numpy array, (62, 92), dtype('float64')
    infile  = datadir('linmod_02_IN.pkl')
    expfile = datadir('linmod_02_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_03():
    # ['Y'] : numpy array, (52, 64, 76), dtype('float64')
    # ['M'] : numpy array, (52, 2), dtype('float64')
    infile  = datadir('linmod_03_IN.pkl')
    expfile = datadir('linmod_03_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_04():
    # ['Y'] : numpy array, (69, 41, 5), dtype('float64')
    # ['M'] : numpy array, (69, 30), dtype('float64')
    infile  = datadir('linmod_04_IN.pkl')
    expfile = datadir('linmod_04_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_05():
    # ['Y'] : numpy array, (81, 1), dtype('float64')
    # ['M'] : numpy array, (81, 2), dtype('float64')
    infile  = datadir('linmod_05_IN.pkl')
    expfile = datadir('linmod_05_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_06():
    # ['Y'] : numpy array, (93, 41, 57), dtype('float64')
    # ['M'] : numpy array, (93, 67), dtype('float64')
    infile  = datadir('linmod_06_IN.pkl')
    expfile = datadir('linmod_06_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_07():
    # ['Y'] : numpy array, (40, 46, 21), dtype('float64')
    # ['M'] : numpy array, (40, 81), dtype('float64')
    infile  = datadir('linmod_07_IN.pkl')
    expfile = datadir('linmod_07_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = SurfStatLinMod(Y, M)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_08():
    # ['Y'] : numpy array, (93, 43), dtype('float64')
    # ['M'] : numpy array, (93, 2), dtype('float64')
    # ['tri'] : numpy array, (93, 3), dtype('int64')
    infile  = datadir('linmod_08_IN.pkl')
    expfile = datadir('linmod_08_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['tri'] = idic['tri']
    slm = SurfStatLinMod(Y, M, surf)
    # oslm : expected output
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_09():
    # ['Y'] : numpy array, (98, 69, 60), dtype('float64')
    # ['M'] : numpy array, (98, 91), dtype('float64')
    # ['tri'] : numpy array, (60, 3), dtype('int64')
    infile  = datadir('linmod_09_IN.pkl')
    expfile = datadir('linmod_09_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['tri'] = idic['tri']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_10():
    # ['Y'] : numpy array, (49, 27), dtype('float64')
    # ['M'] : numpy array, (49, 2), dtype('float64')
    # ['lat'] : numpy array, (3, 3, 3), dtype('bool')
    infile  = datadir('linmod_10_IN.pkl')
    expfile = datadir('linmod_10_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['lat'] = idic['lat']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_11():
    # ['Y'] : numpy array, (45, 27, 3), dtype('float64')
    # ['M'] : numpy array, (45, 7), dtype('float64')
    # ['lat'] : numpy array, (3, 3, 3), dtype('int64')
    infile  = datadir('linmod_11_IN.pkl')
    expfile = datadir('linmod_11_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['lat'] = idic['lat']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_12():
    # ['Y'] : numpy array, (10, 20484), dtype('float64')
    # ['age'] : numpy array, (1, 10), dtype('float64')
    # ['tri'] : numpy array, (40960, 3), dtype('int32')
    # ['coord'] :numpy array, (3, 20484), dtype('float64')
    infile  = datadir('linmod_12_IN.pkl')
    expfile = datadir('linmod_12_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_13():
    # ['Y'] : numpy array, (10, 20484), dtype('float64')
    # ['age'] : numpy array, (1, 10), dtype('float64')
    # ['tri'] : numpy array, (40960, 3), dtype('int32')
    # ['coord'] : numpy array, (3, 20484), dtype('float64')
    infile  = datadir('linmod_13_IN.pkl')
    expfile = datadir('linmod_13_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_14():
    # ['Y'] : numpy array, (10, 20484), dtype('float64')
    # ['age'] : numpy array, (1, 10), dtype('float64')
    # ['tri'] : numpy array, (40960, 3), dtype('int32')
    # ['coord'] : numpy array, (3, 20484), dtype('float64')
    infile  = datadir('linmod_14_IN.pkl')
    expfile = datadir('linmod_14_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_15():
    # building the model term while testing..
    # ['Y'] : numpy array, (20, 20484), dtype('float64')
    # ['params'] : numpy array, (20, 9), dtype('uint16')
    # ['colnames'] : numpy array, (9,), dtype('<U11')
    # ['tri'] : numpy array, (40960, 3), dtype('int32')
    # ['coord'] : numpy array, (3, 20484), dtype('float64')
    infile  = datadir('linmod_15_IN.pkl')
    expfile = datadir('linmod_15_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y        = idic['Y']
    params   = idic['params']
    colnames = list(idic['colnames'])
    M        = Term(params, colnames)
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_16():
    # ['Y'] : numpy array, (20, 20484), dtype('float64')
    # ['params'] : numpy array, (20, 9), dtype('uint16')
    # ['tri'] : numpy array, (40960, 3), dtype('int32')
    # ['coord'] :numpy array, (3, 20484), dtype('float64')
    infile  = datadir('linmod_16_IN.pkl')
    expfile = datadir('linmod_16_OUT.pkl')
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()
    Y        = idic['Y']
    params   = idic['params']
    M        = Term(params)
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)
    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


