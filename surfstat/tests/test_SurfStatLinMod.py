import sys
sys.path.append("python")
from SurfStatLinMod import *
import surfstat_wrap as sw
import numpy as np
from term import Term

sw.matlab_init_surfstat()

def dummy_test(A, B):

    try:
        # wrap matlab functions
        # For term inputs, we can't put these directly into the MATLAB engine.
        if isinstance(A,Term) or isinstance(B,Term):
            if isinstance(A,Term):
                Amat = A.matrix.values
            if isinstance(B,Term):
                Bmat = B.matrix.values
            Wrapped_SurfStatLinMod, bla_tmp = sw.matlab_SurfStatLinMod(Amat, Bmat)
        else:
            Wrapped_SurfStatLinMod, bla_tmp = sw.matlab_SurfStatLinMod(A, B)
    except:
        pytest.fail("ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS...")
	
    # run python functions
    Python_SurfStatLinMod = py_SurfStatLinMod(A, B)

    # compare matlab-python outputs
    testout_SurfStatLinMod = []

    for key in Wrapped_SurfStatLinMod:
        testout_SurfStatLinMod.append(np.allclose(Python_SurfStatLinMod[key], \
                                      Wrapped_SurfStatLinMod[key], \
                                      rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_SurfStatLinMod)


#### test 1a
# 1D inputs --- row vectors
def test_1d_row_vectors():
    v = np.random.randint(1,100)

    A = np.random.rand(1,v)
    B = np.random.rand(1,v)

    dummy_test(A, B)

#### test 2a
# 2D inputs --- square matrices
def test_2d_square_matrices():
    n = np.random.randint(1,100)

    A = np.random.rand(n,n)
    B = np.random.rand(n,n)

    dummy_test(A, B)

#### test 2b
# 2D inputs --- rectangular matrices
def test_2d_rectangular_matrices():
    n = np.random.randint(1,100)
    p = np.random.randint(1,100)
    v = np.random.randint(1,100)

    A = np.random.rand(n,v)
    B = np.random.rand(n,p)

    dummy_test(A, B)


#### test 3
# 3D inputs --- A is a 3D input, B is 1D
def test_3d_A_is_3d_B_is_1d():
    n = np.random.randint(1,100)
    p = np.random.randint(1,100)
    k = np.random.randint(1,100)
    v = np.random.randint(1,100)

    A = np.random.rand(n,v,k)
    B = np.random.rand(n,1)

    dummy_test(A, B)

#### test 3
# 3D inputs --- A is a 3D input, B is 2D
def test_3d_A_is_3d_B_is_2d():
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    v = np.random.randint(1,100)
    p = np.random.randint(1,100)

    A = np.random.rand(n,v,k)
    B = np.random.rand(n,p)

    dummy_test(A, B)


def test_1d_column_vectors():

    v = np.random.randint(1,100)

    A = np.random.rand(v,1)
    B = np.random.rand(v,1)

    dummy_test(A, B)

# 1D terms
def test_1d_terms():

    v = np.random.randint(1,100)

    A = Term(np.random.rand(1,v))
    B = Term(np.random.rand(1,v))

    dummy_test(A, B)