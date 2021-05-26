import numpy as np
import pickle
from testutil import datadir
from nilearn import datasets
from brainstat.mesh.data import mesh_normalize
from sklearn.model_selection import ParameterGrid
from brainstat.context.utils import read_surface_gz
from brainspace.mesh.mesh_elements import get_cells
from brainstat.stats._multiple_comparisons import stat_threshold


def generate_stat_threshold_out(I):
    
    D = {}
    D["peak_threshold"], D["extent_threshold"], D["peak_threshold_1"], \
    D["extent_threshold_1"], D["t"], \
    D["rho"] = stat_threshold(I["search_volume"],
                              I["num_voxels"],
                              I["fwhm"],
                              I["df"],
                              I["p_val_peak"],
                              I["cluster_threshold"],
                              I["p_val_extent"],
                              I["nconj"],
                              I["nvar"],
                              None,
                              None,
                              I["nprint"])
    return D

def params2files(I, D, test_num):
    """Converts params to input/output files"""
    basename = "xthresh"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)
    return

# search_volume is a float, a list, 1D array

np.random.seed(0)

mygrid = [
{'search_volume': [np.random.rand()*5,
                   np.random.randint(1,20, size=(3,)).tolist(),],
                   #np.random.rand(3,)*10,
                   #np.random.rand(2,2)*20],
 'num_voxels': [int(1), 
                #np.random.randint(1,500),
                np.random.randint(1,10, size=(3,)).tolist(),],
                #np.random.rand(5,),
                #np.random.rand(23,1)],
                
 'fwhm': [0.0, np.random.rand()*5],
 'df': [5, np.random.randint(10,size=(2,2))],
 'p_val_peak': [0.05, 
                np.random.rand(4,).tolist(),],
                #np.random.rand(7,),
                #np.random.rand(5,5)],
 'cluster_threshold': [0.001],
 'p_val_extent': [0.05],
 'nconj': [0.5],
 'nvar': [1],
 'nprint': [0]}

]

myparamgrid = ParameterGrid(mygrid)


test_num = 0
for params in myparamgrid:
    test_num += 1
    I = {}
    for key in params.keys():
        I[key] = params[key]
    D = generate_stat_threshold_out(I)
    params2files(I,D,test_num)
    print('AAAA ', test_num)
