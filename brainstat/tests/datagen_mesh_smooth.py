"""Data generation for mesh_smooth unit tests."""

import pickle

import numpy as np
from brainspace.mesh.mesh_elements import get_cells
from nilearn import datasets

from brainstat.context.utils import read_surface_gz
from brainstat.mesh.data import mesh_smooth
from brainstat.tests.testutil import datadir


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


# Single test with real data.
# There's no forks in this function other than the tri/lat which is already
# tested through the mesh_edges tests. We have to use real data as random data
# is likely to result in nan arrays due to faulty fake meshes.
def generate_tests():
    pial_fs5 = datasets.fetch_surf_fsaverage()["pial_left"]
    pial_surf = read_surface_gz(pial_fs5)
    tri = np.array(get_cells(pial_surf)) + 1

    np.random.seed(0)
    data = {"tri": tri, "Y": np.random.uniform(-1, 1, (1, 10242)), "FWHM": 3}
    O = generate_smooth_out(data)
    params2files(data, O, 1)


if __name__ == "__main__":
    generate_tests()
