import numpy as np
import pickle
from testutil import datadir
from brainstat.mesh.utils import mesh_edges


def generate_random_mesh_edge_data(
    key_dim,
    finname,
    key_name="tri",
    key_dtype="float",
    seed=0,
):
    """ Generate random test datasets. """
    # key_dim : tuple, dimensions of array
    # finname : string, filename ending with *pkl
    # key_name : string, "tri" or "lat"
    # key_dtype : string, float or int

    np.random.seed(seed=seed)

    surf = {}

    if key_name == "tri":
        if key_dtype == "float":
            surf["tri"] = np.random.random_sample(key_dim)
        else:
            surf["tri"] = np.random.randint(0, 1024, size=key_dim)
            surf["tri"] = np.array(surf["tri"], dtype=key_dtype)

    if key_name == "lat":
        surf["lat"] = np.random.randint(0, 2, size=key_dim)
        surf["lat"] = np.array(surf["lat"], dtype=key_dtype)

    with open(finname, "wb") as handle:
        pickle.dump(surf, handle, protocol=4)
    return surf


def get_meshedge_output(surf, foutname):
    """ Runs mesh_edges and returns all relevant output. """

    # run mesh_edges
    surf_out = {}
    surf_out["edg"] = mesh_edges(surf)

    with open(foutname, "wb") as handle:
        pickle.dump(surf_out, handle, protocol=4)  #

    return


def generate_data_test_mesh_edges():

    ### test_01 data in-out generation
    print("test_mesh_edges.py : test_01 data is generated..")
    # ['tri'] is a 2D numpy array of shape (78, 3), dtype('float64')
    key_dim = (78, 3)
    finname = datadir("xstatedg_01_IN.pkl")
    key_name = "tri"
    key_dtype = "float"
    seed = 444
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_01_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_02 data in-out generation
    print("test_mesh_edges.py : test_02 data is generated..")
    # ['lat'] is a 2D numpy array of shape (10, 10), dtype('float64')
    key_dim = (10, 10)
    finname = datadir("xstatedg_02_IN.pkl")
    key_name = "lat"
    key_dtype = "float"
    seed = 445
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_02_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_03 data in-out generation
    print("test_mesh_edges.py : test_03 data is generated..")
    # ['lat'] is a 3D numpy array of shape (10, 10, 10), dtype('int64')
    key_dim = (10, 10, 10)
    finname = datadir("xstatedg_03_IN.pkl")
    key_name = "lat"
    key_dtype = "int64"
    seed = 446
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_03_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_04 data in-out generation
    print("test_mesh_edges.py : test_04 data is generated..")
    # ['tri'] is a 2D numpy array of shape (2044, 3), dtype('uint16')
    key_dim = (2044, 3)
    finname = datadir("xstatedg_04_IN.pkl")
    key_name = "tri"
    key_dtype = "uint16"
    seed = 447
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_04_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_05 data in-out generation
    print("test_mesh_edges.py : test_05 data loaded from thicknes_n10.pkl..")
    # ['tri'] is a 2D numpy array of shape (40960, 3), from "thickness_n10.pkl"
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    D = pickle.load(ifile)
    ifile.close()
    surf = {}
    surf["tri"] = D["tri"]
    foutname = datadir("xstatedg_05_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_06 data in-out generation (real data shuffled)
    print("test_mesh_edges.py : test_06 data is generated")
    # test 05 thickiness_n10 data is shuffled
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    D = pickle.load(ifile)
    ifile.close()
    surf = {}
    surf["tri"] = D["tri"]
    np.random.seed(seed=448)
    np.random.shuffle(surf["tri"])
    finname = datadir("xstatedg_06_IN.pkl")
    with open(finname, "wb") as handle:
        pickle.dump(D, handle, protocol=4)
    foutname = datadir("xstatedg_06_OUT.pkl")
    get_meshedge_output(surf, foutname)
