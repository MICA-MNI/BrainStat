import numpy as np
import pickle
from sklearn.model_selection import ParameterGrid
from nilearn import datasets
from brainspace.mesh.mesh_elements import get_cells, get_edges, get_points
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainstat.stats._multiple_comparisons import peak_clus
from brainstat.mesh.utils import mesh_edges
from brainstat.context.utils import read_surface_gz
from testutil import generate_slm, save_slm, datadir, slm2dict


def generate_random_peak_clus(
    surf,
    n_var=1,
    dfs=None,
    cluster_threshold=0.001,
    reselspvert=False,
    edg=False,
):
    """Generates a valid SLM for a surface.
    Parameters
    ----------
    surf : BSPolyData or a dictionary with key 'tri'
        Brain surface.
    n_var : int, optional
        slm.k, by default 1.
    dfs : np.array, None, optional
        Effective degrees of freedom, by default None.
    cluster_threshold : float, optional
        Cluster threshold, by default 0.001.
    reselspvert : bool, optional
        Resels per vertex.
    edg: bool, optional

    Returns
    -------
    brainstat.stats.SLM
        SLM object.
    reselspvert
    edg
    """

    if isinstance(surf, BSPolyData):
        triangles = np.array(get_cells(surf))
        edges = get_edges(surf)
        vertices = get_points(surf)
        n_vertices = vertices.shape[0]
        n_edges = edges.shape[0]
    else:
        triangles = surf["tri"]
        edges = mesh_edges(surf)
        n_edges = edges.shape[0]
        n_vertices = int(surf["tri"].shape[0] / 2)

    if dfs is not None:
        dfs = np.random.randint(1, 100, (1, n_vertices))

    slm = generate_slm(
        t=np.random.random_sample((1, n_vertices)),
        df=np.random.randint(2, 100),
        k=n_var,
        resl=np.random.random_sample((n_edges, 1)),
        tri=triangles + 1,
        surf=surf,
        dfs=dfs,
        mask=np.random.choice(a=[False, True], size=(n_vertices)),
        cluster_threshold=cluster_threshold,
    )

    thresh = np.random.rand()

    if reselspvert:
        reselspvert = np.random.rand(
            n_vertices,
        )
    else:
        reselspvert = None

    if edg:
        edg = edges
    else:
        edg = None

    return slm, thresh, reselspvert, edg


def params2files(slm, thresh, reselspvert, edg, basename, test_num):
    """Converts an params input/output files"""
    I = {}
    I["t"] = slm.t
    I["tri"] = slm.tri
    I["mask"] = slm.mask
    I["thresh"] = thresh
    I["reselspvert"] = reselspvert
    I["edg"] = edg
    I["k"] = slm.k
    I["df"] = slm.df
    I["dfs"] = slm.dfs

    D = {}
    D["peak"], D["clus"], D["clusid"] = peak_clus(
        slm, thresh, reselspvert=reselspvert, edg=edg
    )
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    print("test data %s for peak_clus generated... " % (str(test_num)))

    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)

    return


basename = "xstatpeakc"
test_num = 0
np.random.seed(0)

# test with surf['tri']
rand_surf = {}
rand_surf["tri"] = np.random.randint(0, int(100 / 2), size=(100, 3))

# iterate the variables
var_types = {
    "t": [np.float64],
    "df": [int, np.uint16],
    "k": [int, np.uint8],
    "resl": [np.float64],
    "tri": [np.int64],
    "mask": [np.int64],
}

type_parameter_grid = ParameterGrid(var_types)

for params in type_parameter_grid:
    slm, thresh, reselspvert, edg = generate_random_peak_clus(rand_surf)
    for key in list(params.keys()):
        attr = getattr(slm, key)
        setattr(slm, key, params[key](attr))
    test_num += 1
    params2files(slm, thresh, reselspvert, edg, basename, test_num)


for params in type_parameter_grid:
    slm, thresh, reselspvert, edg = generate_random_peak_clus(
        rand_surf, reselspvert=True, edg=True
    )
    for key in list(params.keys()):
        attr = getattr(slm, key)
        setattr(slm, key, params[key](attr))
    test_num += 1
    params2files(slm, thresh, reselspvert, edg, basename, test_num)


for params in type_parameter_grid:
    slm, thresh, reselspvert, edg = generate_random_peak_clus(
        rand_surf, reselspvert=True, edg=True
    )
    for key in list(params.keys()):
        attr = getattr(slm, key)
        setattr(slm, key, params[key](attr))
    test_num += 1
    params2files(slm, thresh, reselspvert, edg, basename, test_num)


# test with surf BSPolydata
pial_fs5 = datasets.fetch_surf_fsaverage()["pial_left"]
pial_surf = read_surface_gz(pial_fs5)
n_vertices = get_points(pial_surf).shape[0]

# Variable type tests
for params in type_parameter_grid:
    slm, thresh, reselspvert, edg = generate_random_peak_clus(pial_surf)
    for key in list(params.keys()):
        attr = getattr(slm, key)
        setattr(slm, key, params[key](attr))
    test_num += 1
    params2files(slm, thresh, reselspvert, edg, basename, test_num)

for params in type_parameter_grid:
    slm, thresh, reselspvert, edg = generate_random_peak_clus(
        pial_surf, reselspvert=True
    )
    for key in list(params.keys()):
        attr = getattr(slm, key)
        setattr(slm, key, params[key](attr))
    test_num += 1
    params2files(slm, thresh, reselspvert, edg, basename, test_num)

# Optional variable test.
var_optional = {
    "dfs": [None, np.int],
    "k": [1, 3],
}


# Additional variable tests.
additional_parameter_grid = ParameterGrid(var_optional)
for params in additional_parameter_grid:
    slm, thresh, reselspvert, edg = generate_random_peak_clus(
        pial_surf, reselspvert=True, edg=False
    )
    for key in list(params.keys()):
        setattr(slm, key, params[key])
    test_num += 1
    params2files(slm, thresh, reselspvert, edg, basename, test_num)
