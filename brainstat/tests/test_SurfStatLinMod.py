import numpy as np
import pytest
from brainstat.stats.LinMod import LinMod
import surfstat_wrap as sw
from brainstat.stats.term import Term
from brainspace.datasets import load_conte69
from scipy.io import loadmat

surfstat_eng = sw.matlab_init_surfstat()


def dummy_test(Y, model, surf=None, resl_check=True):

    py_slm = LinMod(Y, model, surf=surf)
    mat_slm = sw.matlab_LinMod(Y, model, surf=surf)

    if not resl_check:
        py_slm['resl'] = np.array([])
        mat_slm['resl'] = np.array([])

    for k in set.union(set(py_slm.keys()), set(mat_slm.keys())):
        assert k in mat_slm, "'%s' missing from MATLAB slm." % k
        assert k in py_slm, "'%s' missing from Python slm." % k

        if k not in ['df', 'dr']:
            assert mat_slm[k].shape == py_slm[k].shape, \
                "Different shape: %s" % k

        assert np.allclose(mat_slm[k], py_slm[k], rtol=1e-05, equal_nan=True), "Not equal: %s" % k


# 2D inputs --- square matrices
def test_01():
    n = np.random.randint(10, 100)

    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    B[:,0] = 1 # Constant term.

    dummy_test(A, B, surf=None)


# 2D inputs --- rectangular matrices
def test_02():
    n = np.random.randint(1, 100)
    p = np.random.randint(1, 100)
    v = np.random.randint(1, 100)

    A = np.random.rand(n, v)
    B = np.random.rand(n, p)
    B[:,0] = 1 # Constant term.

    dummy_test(A, B, surf=None)


# 3D inputs --- A is a 3D input, B is 1D
def test_03():
    n = np.random.randint(1, 100)
    k = np.random.randint(2, 100)
    v = np.random.randint(1, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n, 2)
    B[:,0] = 1 # Constant term.

    dummy_test(A, B, surf=None)


# 3D inputs --- A is a 3D input, B is 2D
def test_04():
    n = np.random.randint(1, 100)
    k = np.random.randint(2, 100)
    v = np.random.randint(1, 100)
    p = np.random.randint(2, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n, p)
    B[:,0] = 1 # Constant term.

    dummy_test(A, B, surf=None)


def test_05():

    v = np.random.randint(10, 100)

    A = np.random.rand(v, 1)
    B = np.random.rand(v, 2)
    B[:,0] = 1 # Constant term.

    dummy_test(A, B, surf=None)


# 1D terms
def test_06():

    n = np.random.randint(10, 100)
    p = np.random.randint(1, 10)

    A = np.random.rand(n, p)
    B = np.random.rand(n,2)
    B[:,0] = 1 # Constant term.
    B = Term(B)

    dummy_test(A, B, surf=None)


# 3D inputs --- A is a 3D input, B is Term
def test_07():
    n = np.random.randint(3, 100)
    k = np.random.randint(3, 100)
    v = np.random.randint(3, 100)
    p = np.random.randint(3, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n,p)
    B[:,0] = 1 # Constant term.
    B = Term(B)

    dummy_test(A, B, surf=None)


# ?
def test_08():
    n = np.random.randint(2, 100)
    v = np.random.randint(2, 100)

    A = np.random.rand(n, v)
    B = np.random.rand(n, 2)
    B[:,0] = 1 # Constant term.

    surf = {'tri': np.random.randint(1, v, size=(n, 3))}
    dummy_test(A, B, surf, resl_check=False)


# 3D inputs --- A is a 3D input, B is Term
def test_09():
    n = np.random.randint(3, 100)
    k = np.random.randint(3, 100)
    v = np.random.randint(3, 100)
    p = np.random.randint(3, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n,p)
    B[:,0] = 1 # Constant term.
    B = Term(B)

    surf = {'tri': np.random.randint(1, v, size=(k, 3))}
    dummy_test(A, B, surf=surf, resl_check=False)


def test_10():
    n = np.random.randint(2, 100)
    v = np.random.randint(27, 28)

    A = np.random.rand(n, v)
    B = np.random.rand(n, 2)
    B[:,0] = 1 # Constant term.

    surf = {'lat': np.random.choice([0, 1], size=(3, 3, 3)).astype(bool)}
    dummy_test(A, B, surf=surf)


# 3D inputs --- A is a 3D input, B is Term
def test_11():
    n = np.random.randint(3, 100)
    k = np.random.randint(3, 10)
    v = np.random.randint(27, 28)
    p = np.random.randint(3, 10)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n,p)
    B[:,0] = 1 # Constant term.
    B = Term(B)

    surf = {'lat': np.random.choice([0, 1], size=(3, 3, 3))}
    dummy_test(A, B, surf, resl_check=False)


def test_12():
    surf, _ = load_conte69()

    p = np.random.randint(1, 10)
    n = np.random.randint(2, 10)

    A = np.random.rand(n,32492)
    B = np.random.rand(n,p)
    B[:,0] = 1 # Constant term.
    B = Term(B)

    dummy_test(A, B, surf, resl_check=False)


# real thickness data for 10 subjects
def test_13():
    fname = './tests/data/thickness.mat'
    f = loadmat(fname)

    A = f['T']
    AGE = Term(np.array(f['AGE']), 'AGE')
    B = 1 + AGE
    surf = {}
    surf['tri'] = f['tri']
    surf['coord'] = f['coord']
    dummy_test(A, B, surf)


# real thickness data for 10 subjects --> shuffle "Y" values
def test_14():
    fname = './tests/data/thickness.mat'
    f = loadmat(fname)

    A = f['T']
    np.random.shuffle(A)

    AGE = Term(np.array(f['AGE']), 'AGE')
    B = 1 + AGE
    surf = {}
    surf['tri'] = f['tri']
    surf['coord'] = f['coord']
    dummy_test(A, B, surf)


# real thickness data for 10 subjects --> shuffle "Y" values, shuffle surf['tri']
def test_15():
    fname = './tests/data/thickness.mat'
    f = loadmat(fname)

    A = f['T']
    np.random.shuffle(A)

    AGE = Term(np.array(f['AGE']), 'AGE')
    B = 1 + AGE
    surf = {}
    surf['tri'] = f['tri']

    np.random.shuffle(surf['tri'])

    surf['coord'] = f['coord']
    dummy_test(A, B, surf)


# real data from sofopofo
def test_16():
    fname = './tests/data/sofopofo1.mat'
    f = loadmat(fname)
    T = f['sofie']['T'][0,0]

    params = f['sofie']['model'][0,0]
    colnames = ['1', 'ak', 'female', 'male', 'Affect', 'Control1', 'Perspective',
    'Presence', 'ink']

    M = Term(params, colnames)

    SW = {}
    SW['tri'] = f['sofie']['SW'][0,0]['tri'][0,0]
    SW['coord'] = f['sofie']['SW'][0,0]['coord'][0,0]

    dummy_test(T, M, SW)


# real data from sofopofo, no column naming in the model Term
def test_17():
    fname = './tests/data/sofopofo1.mat'
    f = loadmat(fname)
    T = f['sofie']['T'][0,0]

    params = f['sofie']['model'][0,0]

    M = Term(params)

    SW = {}
    SW['tri'] = f['sofie']['SW'][0,0]['tri'][0,0]
    SW['coord'] = f['sofie']['SW'][0,0]['coord'][0,0]

    dummy_test(T, M, SW)
