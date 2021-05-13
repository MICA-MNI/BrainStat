import numpy as np
import pickle
from sklearn.model_selection import ParameterGrid
from brainstat.mesh.data import mesh_smooth
from brainspace.mesh.mesh_elements import get_cells
from brainstat.context.utils import read_surface_gz
from nilearn import datasets
from testutil import datadir


def generate_smooth_out(rand_dict):
    """Uses rand_dict to run mesh_smooth and returns the smoothed data."""
    # below are going to be input params for mesh_smooth
    Y = rand_dict["Y"]
    FWHM = rand_dict["FWHM"]
    surf = {}
    if "tri" in rand_dict.keys():
        surf["tri"] = rand_dict["tri"]
    if "lat" in rand_dict.keys():
        surf["lat"] = rand_dict["lat"]

    # run mesh_smooth and return the smoothed data
    O = {}
    O["Python_Y"] = mesh_smooth(Y, surf, FWHM)

    return O


def params2files(rand_dict, O, test_num):
    """Converts params to input/output files"""
    # filenames for the input and outpur dict's
    basename = "xstatsmo"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")

    # save input and output data in pickle format with filenames above
    with open(fin_name, "wb") as f:
        pickle.dump(rand_dict, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(O, g, protocol=4)
    return


# test data with small random stuff
# Test 1: Y square 2D array, FWHM in range [0, 1], tri (20,3)
# Test 2: Y 3D array, FWHM in range [0, 1], tri (20,3)
# Test 3: Y square 2D array, FWHM int(3), tri (20,3)
# Test 4: Y 3D array, FWHM int(3), tri (20,3)
# Test 5: Y square 2D array, FWHM in range [0, 1], lat (3,3,3)
# Test 6: Y 3D array, FWHM in range [0, 1], lat (3,3,3)
# Test 7: Y square 2D array, FWHM int(3), lat (3,3,3)
# Test 8: Y 3D array, FWHM int(3), lat (3,3,3)

np.random.seed(0)
mygrid = [
    {
        "Y": [np.random.rand(72, 72), np.random.rand(30, 30, 10)],
        "FWHM": [np.random.rand(), int(3.0)],
        "tri": [np.random.randint(0, 10, size=(20, 3))],
    },
    {
        "Y": [np.random.rand(72, 72), np.random.rand(30, 30, 10)],
        "FWHM": [np.random.rand(), int(3.0)],
        "lat": [np.random.choice(a=[1, 0], size=(3, 3, 3))],
    },
]

myparamgrid = ParameterGrid(mygrid)

test_num = 0
for params in myparamgrid:
    rand_dict = {}
    for key in list(params.keys()):
        rand_dict[key] = params[key]
    O = generate_smooth_out(rand_dict)
    test_num += 1
    params2files(rand_dict, O, test_num)

# test data with real triangle coordinates
# Test 9: Y 2D random array (1,10242), FWHM float [0,1], tri pial_fs5
# Test 10: Y 2D random array (1,10242), FWHM float [0,1], tri pial_fs5 shuffled
# Test 11: Y 2D random array (1,10242), FWHM int, tri pial_fs5
# Test 12: Y 2D random array (1,10242), FWHM int, tri pial_fs5 shuffled

pial_fs5 = datasets.fetch_surf_fsaverage()["pial_left"]
pial_surf = read_surface_gz(pial_fs5)
real_tri = np.array(get_cells(pial_surf))

np.random.seed(0)
real_tri_copy = real_tri.copy()
np.random.shuffle(real_tri_copy)

mygrid_realtri = [
    {
        "Y": [np.random.uniform(-1, 1, (1, 10242))],
        "FWHM": [np.random.rand(), np.random.randint(10)],
        "tri": [real_tri, real_tri_copy],
    }
]

myparamgridreal = ParameterGrid(mygrid_realtri)

for params in myparamgridreal:
    rand_dict = {}
    for key in list(params.keys()):
        rand_dict[key] = params[key]
    O = generate_smooth_out(rand_dict)
    test_num += 1
    params2files(rand_dict, O, test_num)
