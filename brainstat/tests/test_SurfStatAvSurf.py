import sys
sys.path.append("python")
import numpy as np
import pytest
from SurfStatAvSurf import py_SurfStatAvSurf
import surfstat_wrap as sw
from brainspace.datasets import load_conte69
from brainspace.mesh.mesh_elements import get_points, get_cells
from brainspace.mesh.mesh_io import write_surface
import tempfile
from itertools import chain

surfstat_eng = sw.matlab_init_surfstat()


# Test function
def dummy_test(py_surfaces, fun = np.add):
    # Run functions
    mat_surf = sw.matlab_SurfStatAvSurf(py_surfaces, fun)
    py_out = py_SurfStatAvSurf(py_surfaces, fun)
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
    for s in surfaces:
        t.append(tempfile.NamedTemporaryFile(suffix='.fs', delete=True))
        names.append(t[-1].name)
        write_surface(s, names[-1], otype='fs')
    return t, names


def test_01():
    # Two surfaces, column vector.
    surfaces = load_conte69()
    t, names = temp_surfaces(surfaces)
    namesArr = np.array(names, ndmin=2).T
    dummy_test(namesArr)
    for i in range(0,len(t)):
        t[i].close()


def test_02():
    # Two surfaces, row vector.
    surfaces = load_conte69()
    t, names = temp_surfaces(surfaces)
    namesArr = np.array(names, ndmin=2)
    dummy_test(namesArr)
    for i in range(0,len(t)):
        t[i].close()


def test_03():
    # Four surfaces, 2-by-2 matrix.
    surfaces_1 = load_conte69()
    surfaces_2 = load_conte69(as_sphere=True)
    t, names = temp_surfaces(surfaces_1 + surfaces_2)
    namesArr = np.reshape(np.array(names, ndmin=2),(2,2))
    dummy_test(namesArr)
    for i in range(0,len(t)):
        t[i].close()


def test_04():
    # Four surfaces, 2-by-2 matrix, minimum.
    surfaces_1 = load_conte69()
    surfaces_2 = load_conte69(as_sphere=True)
    t, names = temp_surfaces(surfaces_1 + surfaces_2)
    namesArr = np.reshape(np.array(names, ndmin=2),(2,2))
    dummy_test(namesArr, np.fmin)
    for i in range(0,len(t)):
        t[i].close()


def test_05():
    # Four surfaces, 2-by-2 matrix, maximum.
    surfaces_1 = load_conte69()
    surfaces_2 = load_conte69(as_sphere=True)
    t, names = temp_surfaces(surfaces_1 + surfaces_2)
    namesArr = np.reshape(np.array(names, ndmin=2),(2,2))
    dummy_test(namesArr, np.fmax)
    for i in range(0,len(t)):
        t[i].close()


def test_06():
    # Six surfaces, 2-by-2 matrix, maximum.
    surfaces_1 = load_conte69()
    surfaces_2 = load_conte69(as_sphere=True)
    surfaces_3 = load_conte69(as_sphere=True)
    t, names = temp_surfaces(surfaces_1 + surfaces_2 + surfaces_3)
    namesArr = np.reshape(np.array(names, ndmin=2),(3,2))
    dummy_test(namesArr, np.fmax)
    for i in range(0,len(t)):
        t[i].close()
