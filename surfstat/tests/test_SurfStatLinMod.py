import sys
sys.path.append("python")
import numpy as np
import pytest
from SurfStatLinMod import py_SurfStatLinMod
import surfstat_wrap as sw
from term import Term

surfstat_eng = sw.matlab_init_surfstat()

def dummy_test(Y, model, surf=None):
    py_slm = py_SurfStatLinMod(Y, model, surf=surf)
    mat_slm = sw.matlab_SurfStatLinMod(Y, model, surf=surf)

    for k in set.union(set(py_slm.keys()), set(mat_slm.keys())):
        assert k in mat_slm, "'%s' missing from MATLAB slm." % k
        assert k in py_slm, "'%s' missing from Python slm." % k

        if k not in ['df', 'dr']:
            assert mat_slm[k].shape == py_slm[k].shape, \
                "Different shape: %s" % k

        assert np.allclose(mat_slm[k], py_slm[k]), "Not equal: %s" % k


# 2D inputs --- square matrices
def test_2d_square_matrices():
    n = np.random.randint(10, 100)

    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    B[:,0] = 1 # Constant term. 
    
    dummy_test(A, B, surf=None)


# 2D inputs --- rectangular matrices
def test_2d_rectangular_matrices():
    n = np.random.randint(1, 100)
    p = np.random.randint(1, 100)
    v = np.random.randint(1, 100)

    A = np.random.rand(n, v)
    B = np.random.rand(n, p)
    B[:,0] = 1 # Constant term. 
    
    dummy_test(A, B, surf=None)


# 3D inputs --- A is a 3D input, B is 1D
def test_3d_A_is_3d_B_is_1d():
    n = np.random.randint(1, 100)
    k = np.random.randint(2, 100)
    v = np.random.randint(1, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n, 2)
    B[:,0] = 1 # Constant term. 
    
    dummy_test(A, B, surf=None)


# 3D inputs --- A is a 3D input, B is 2D
def test_3d_A_is_3d_B_is_2d():
    n = np.random.randint(1, 100)
    k = np.random.randint(2, 100)
    v = np.random.randint(1, 100)
    p = np.random.randint(2, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n, p)
    B[:,0] = 1 # Constant term. 
    
    dummy_test(A, B, surf=None)


def test_1d_column_vectors():

    v = np.random.randint(10, 100)

    A = np.random.rand(v, 1)
    B = np.random.rand(v, 2)
    B[:,0] = 1 # Constant term. 
    
    dummy_test(A, B, surf=None)


# 1D terms
def test_1d_terms():

    n = np.random.randint(10, 100)
    p = np.random.randint(1, 10)

    A = np.random.rand(n, p)
    B = np.random.rand(n,2)
    B[:,0] = 1 # Constant term. 
    B = Term(B)  
    
    dummy_test(A, B, surf=None)


# 3D inputs --- A is a 3D input, B is Term
def test_3d_A_is_3d_B_is_term():
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
def test_2d_A_is_1d_B_is_surf_tri():
    n = np.random.randint(2, 100)
    v = np.random.randint(2, 100)

    A = np.random.rand(n, v)
    B = np.random.rand(n, 2)
    B[:,0] = 1 # Constant term. 
    
    surf = {'tri': np.random.randint(1, v, size=(n, 3))}
    dummy_test(A, B, surf)


# 3D inputs --- A is a 3D input, B is Term
def test_3d_A_is_2d_B_is_term_surf_tri():
    n = np.random.randint(3, 100)
    k = np.random.randint(3, 100)
    v = np.random.randint(3, 100)
    p = np.random.randint(3, 100)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n,p)
    B[:,0] = 1 # Constant term. 
    B = Term(B)  
    
    surf = {'tri': np.random.randint(1, v, size=(k, 3))}
    dummy_test(A, B, surf=surf)


def test_2d_A_is_1d_B_is_surf_lat():
    n = np.random.randint(2, 100)
    v = np.random.randint(27, 28)

    A = np.random.rand(n, v)
    B = np.random.rand(n, 2)
    B[:,0] = 1 # Constant term. 
    
    surf = {'lat': np.random.choice([0, 1], size=(3, 3, 3)).astype(bool)}
    dummy_test(A, B, surf=surf)


# 3D inputs --- A is a 3D input, B is Term
def test_3d_A_is_2d_B_is_term_surf_lat():
    n = np.random.randint(3, 100)
    k = np.random.randint(3, 10)
    v = np.random.randint(27, 28)
    p = np.random.randint(3, 10)

    A = np.random.rand(n, v, k)
    B = np.random.rand(n,p)
    B[:,0] = 1 # Constant term. 
    B = Term(B)  
    
    surf = {'lat': np.random.choice([0, 1], size=(3, 3, 3))}
    dummy_test(A, B, surf)
