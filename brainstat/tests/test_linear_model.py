import numpy as np
from .testutil import datadir
from brainstat.stats.terms import Term
from brainstat.stats.SLM import SLM
import h5py


def dummy_test(infile, expfile, simple=True):

    testout = []

    # infile includes at least 'Y'
    hin = h5py.File(infile,'r')
    Y = np.array(hin['Y'])

    # if simple, then, key 'M' is given
    if simple:
        M = np.array(hin['M'])
    # otherwise, key 'AGE' is given
    else:
        M = np.array(hin['AGE'])
        AGE = Term(M, "AGE")
        M = 1 + AGE

    slm = SLM(M, Term(1))

    # infile might include 'tri', 'coord', 'lat' keys
    for makey in hin.keys():
        if makey == 'tri':
            slm.surf = {'tri': np.array(hin['tri'])}
        if makey == 'lat':
            slm.surf = {'lat': np.array(hin['lat'])}
        if makey == 'coord':
            slm.surf = {'coord': np.array(hin['coord'])}

    # here we go --> run the linear model
    slm.linear_model(Y)

    # expfile includes the expected output of the linear model
    hexp = h5py.File(expfile,'r')

    # compare...
    for makey_ in hexp.keys():
        if makey_ != 'surf':
            comp = np.allclose(getattr(slm, makey_), hexp[makey_],
                               rtol=1e-05, equal_nan=True)
            print(comp)
            testout.append(comp)
    assert all(flag == True for (flag) in testout)


def test_01():
    # ['Y'] and ['M'] small 2D suqare arrays
    # ['Y'] : np array, shape (43, 43), dtype('float64')
    # ['M'] : np array, (43, 43), dtype('float64')
    infile = datadir("linmod_01_IN.h5")
    expfile = datadir("linmod_01_OUT.h5")
    dummy_test(infile, expfile)


def test_02():
    # ['Y'] and ['M'] small 2D rectengular arrays
    # ['Y'] : np array, (62, 7), dtype('float64')
    # ['M'] : np array, (62, 92), dtype('float64')
    infile = datadir("linmod_02_IN.h5")
    expfile = datadir("linmod_02_OUT.h5")
    dummy_test(infile, expfile)


def test_03():
    # ['Y'] is a 3D array, ['M'] is a 2D array
    # ['Y'] : np array, (52, 64, 76), dtype('float64')
    # ['M'] : np array, (52, 2), dtype('float64')
    infile = datadir("linmod_03_IN.h5")
    expfile = datadir("linmod_03_OUT.h5")
    dummy_test(infile, expfile)


def test_04():
    # similar to test_03, shapes of ['Y'] and ['M'] changed
    # ['Y'] : np array, (69, 41, 5), dtype('float64')
    # ['M'] : np array, (69, 30), dtype('float64')
    infile = datadir("linmod_04_IN.h5")
    expfile = datadir("linmod_04_OUT.h5")
    dummy_test(infile, expfile)


def test_05():
    # ['Y'] and ['M'] small 2D rectengular arrays, size(Y) < size(M)
    # ['Y'] : np array, (81, 1), dtype('float64')
    # ['M'] : np array, (81, 2), dtype('float64')
    infile = datadir("linmod_05_IN.h5")
    expfile = datadir("linmod_05_OUT.h5")
    dummy_test(infile, expfile)


def test_06():
    # ['Y'] is a 3D array, ['M'] is a 2D array, M has more columns than Y
    # ['Y'] : np array, (93, 41, 57), dtype('float64')
    # ['M'] : np array, (93, 67), dtype('float64')
    infile = datadir("linmod_06_IN.h5")
    expfile = datadir("linmod_06_OUT.h5")
    dummy_test(infile, expfile)


def test_07():
    # similar to test_06, differently shaped Y and M
    # ['Y'] : np array, (40, 46, 21), dtype('float64')
    # ['M'] : np array, (40, 81), dtype('float64')
    infile = datadir("linmod_07_IN.h5")
    expfile = datadir("linmod_07_OUT.h5")
    dummy_test(infile, expfile)


def test_08():
    # ['Y'] and ['M'] mid. sized 2D arrays + optional ['tri'] input for surf
    # ['Y'] : np array, (93, 43), dtype('float64')
    # ['M'] : np array, (93, 2), dtype('float64')
    # ['tri'] : np array, (93, 3), dtype('int64')
    infile = datadir("linmod_08_IN.h5")
    expfile = datadir("linmod_08_OUT.h5")
    dummy_test(infile, expfile)


def test_09():
    # ['Y'] is 3D array, ['M'] is 2D array, M has more cols than Y, tri given
    # ['Y'] : np array, (98, 69, 60), dtype('float64')
    # ['M'] : np array, (98, 91), dtype('float64')
    # ['tri'] : np array, (60, 3), dtype('int64')
    infile = datadir("linmod_09_IN.h5")
    expfile = datadir("linmod_09_OUT.h5")
    dummy_test(infile, expfile)


def test_10():
    # similar to test_02 + optional ['lat'] input for surf
    # ['Y'] : np array, (49, 27), dtype('float64')
    # ['M'] : np array, (49, 2), dtype('float64')
    # ['lat'] : np array, (3, 3, 3), dtype('int64'), 1's or 0's
    infile = datadir("linmod_10_IN.h5")
    expfile = datadir("linmod_10_OUT.h5")
    dummy_test(infile, expfile)


def test_11():
    # similar to test_03 + optional ['lat'] input for surf
    # ['Y'] : np array, (45, 27, 3), dtype('float64')
    # ['M'] : np array, (45, 7), dtype('float64')
    # ['lat'] : np array, (3, 3, 3), dtype('int64'), 1's or 0's
    infile = datadir("linmod_11_IN.h5")
    expfile = datadir("linmod_11_OUT.h5")
    dummy_test(infile, expfile)


def test_12():
    # real dataset, ['Y'] 20k columns, ['age'] modelling with Term, ['tri'] 40k vertex
    # ['Y'] : np array, (10, 20484), dtype('float64')
    # ['age'] : np array, (1, 10), dtype('float64')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    infile = datadir("thickness_n10.h5")
    expfile = datadir("linmod_12_OUT.h5")
    dummy_test(infile, expfile, simple=False)


def test_13():
    # similar to test_12, ['Y'] values shuffled
    # ['Y'] : np array, (10, 20484), dtype('float64')
    # ['age'] : np array, (1, 10), dtype('float64')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    infile = datadir("linmod_13_IN.h5")
    expfile = datadir("linmod_13_OUT.h5")
    dummy_test(infile, expfile, simple=False)


def test_14():
    # similar to test_12, ['Y'] and ['tri'] values shuffled
    # ['Y'] : np array, (10, 20484), dtype('float64')
    # ['age'] : np array, (1, 10), dtype('float64')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    infile = datadir("linmod_14_IN.h5")
    expfile = datadir("linmod_14_OUT.h5")
    dummy_test(infile, expfile, simple=False)


def test_15():
    # similar to test_12, ['Y'] size doubled + model params extended
    # ['Y'] : np array, (20, 20484), dtype('float64')
    # ['M'] : np array, (20, 9), dtype('uint16')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    infile = datadir("linmod_15_IN.h5")
    expfile = datadir("linmod_15_OUT.h5")
    dummy_test(infile, expfile)
