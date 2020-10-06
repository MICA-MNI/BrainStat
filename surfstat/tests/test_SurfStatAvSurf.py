import sys, os
sys.path.append("python")
import numpy as np
import pytest
from SurfStatAvSurf import py_SurfStatAvSurf
import surfstat_wrap as sw
from brainspace.datasets import load_conte69
from brainspace.mesh.mesh_elements import get_points, get_cells 
from brainspace.mesh.mesh_io import read_surface, write_surface
import tempfile

surfstat_eng = sw.matlab_init_surfstat()

# Test function
def dummy_test(py_surfaces, mat_surfaces, py_fun=lambda x, y: x+y, mat_fun=None, 
               transpose=False, dimensionality=[]):
    # Run functions
    mat_surf = sw.matlab_SurfStatAvSurf(mat_surfaces, mat_fun, transpose, dimensionality)      
    py_out = py_SurfStatAvSurf(py_surfaces, py_fun)   
    py_surf = {'tri': np.array(get_cells(py_out)+1), 
               'coord': np.array(get_points(py_out)).T}
    
    # Sort triangles. 
    py_surf['tri'] = np.sort(py_surf['tri'], axis=1)
    mat_surf['tri'] = np.sort(mat_surf['tri'], axis=1)
    
    # Check equality.
    for k in set.union(set(py_surf.keys()), set(mat_surf.keys())):
        assert k in mat_surf, "'%s' missing from MATLAB slm." % k
        assert k in py_surf, "'%s' missing from Python slm." % k

        if k not in ['df', 'dr']:
            assert mat_surf[k].shape == py_surf[k].shape, \
                "Different shape: %s" % k
        assert np.allclose(mat_surf[k], py_surf[k]), "Not equal: %s" % k

# Write surfaces to temporary files. 
def temp_surfaces(surfaces):
    t = []
    names = []
    default_tmp_dir = tempfile._get_default_tempdir()
    for s in surfaces:
        t.append(tempfile.NamedTemporaryFile(suffix='.fs', delete=True))
        names.append(t[-1].name)
        write_surface(s, names[-1], otype='fs')
    return t, names

## Tests
# Two surfaces, column vector.
def test_two_surfaces_column_plus():
    surfaces = load_conte69()
    t, names = temp_surfaces(surfaces)
    dummy_test(np.array(names, ndmin=2).T, names, transpose=True)
    for i in range(0,len(t)):
        t[i].close()

# Two surfaces, row vector.
def test_two_surfaces_row_plus():
    surfaces = load_conte69()
    t, names = temp_surfaces(surfaces)
    dummy_test(np.array(names, ndmin=2), names, transpose=False)
    for i in range(0,len(t)):
        t[i].close()

# Four surfaces, 2-by-2 matrix.
def test_four_surfaces_square_plus():
    surfaces_1 = load_conte69()
    surfaces_2 = load_conte69(as_sphere=True)
    t, names = temp_surfaces(surfaces_1 + surfaces_2)
    py_surfaces = np.reshape(np.array(names), (2,2))
    dummy_test(np.reshape(np.array(names, ndmin=2),(2,2), order='F'), names, 
               transpose=False, dimensionality=surfstat_eng.cell2mat([2,2]))
    for i in range(0,len(t)):
        t[i].close()

# Four surfaces, 2-by-2 matrix, minimum.
def test_four_surfaces_square_min():
    surfaces_1 = load_conte69()
    surfaces_2 = load_conte69(as_sphere=True)
    t, names = temp_surfaces(surfaces_1 + surfaces_2)
    
    py_surfaces = np.reshape(np.array(names), (2,2))
    py_fun = lambda x,y: np.minimum(x,y)
    mat_fun = surfstat_eng.str2func('min')
    
    dummy_test(np.reshape(np.array(names, ndmin=2),(2,2), order='F'), names, 
               py_fun, mat_fun, transpose=False, dimensionality=surfstat_eng.cell2mat([2,2]))
    for i in range(0,len(t)):
        t[i].close()


