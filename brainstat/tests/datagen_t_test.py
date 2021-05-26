import numpy as np
import pickle
from testutil import datadir
from brainstat.stats._t_test import t_test
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from sklearn.model_selection import ParameterGrid

def generate_t_test_out(I):

    slm = SLM(FixedEffect(1), FixedEffect(1))
    for key in I.keys():
        setattr(slm, key, I[key])
    # run t_test
    t_test(slm)
    expkeys = ['X', 'df', 'coef', 'SSE', 'c', 'k', 'ef', 'sd', 't']
    D = {}
    for key in expkeys:
        D[key] = getattr(slm, key)    
    return D
    
def params2files(I, D, test_num):
    """Converts params to input/output files"""
    basename = "xstatt"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)
    return
    

np.random.seed(0)

# these are the sizes of array, which will be looped in Parameter Grid
mygrid_xy = [{"x":[3],"y":[1]},
             {"x":[6],"y":[6]},
             {"x":[5],"y":[2]} ]
myparamgrid_xy = ParameterGrid(mygrid_xy)

# Here wo go!
test_num = 0
for params_xy in list(myparamgrid_xy):    
    x = params_xy["x"]
    y = params_xy["y"]
    
    mygrid = [
    {
        "X": [np.random.rand(x,y)],
        "df" : [int(y-1)],
        "coef" : [np.random.rand(y,x),],
        "SSE" : [np.random.rand(1,x)],
        "contrast" : [np.random.rand(1,y), np.random.rand(1,1)],
        "dr" : [None, int(y+1)],       
        }]

    # here goes the actual Parameter Grid    
    myparamgrid = ParameterGrid(mygrid)
    for params in myparamgrid:
        test_num += 1
        I = {}
        for key in params.keys():
            if params[key] is not None:
                I[key] = params[key]
        print(test_num)        
        D = generate_t_test_out(I)
        params2files(I, D, test_num)