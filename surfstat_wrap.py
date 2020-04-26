import matlab.engine
import matlab
import os
import numpy as np

def matlab_init_surfstat():
    global surfstat_eng
    surfstat_eng = matlab.engine.start_matlab()
    addpath = surfstat_eng.addpath('surfstat')

def matlab_SurfStatLinMod(T, M):

    if isinstance(T, np.ndarray):
        T = matlab.double(T.tolist())
    else:
        T = surfstat_eng.double(T)

    if isinstance(M, np.ndarray):
        M = matlab.double(M.tolist())
    else:
        M = surfstat_eng.double(M)

    result_mat = surfstat_eng.SurfStatLinMod(T, M)

    result_py = {key: None for key in result_mat.keys()}

    for key in result_mat:
        result_py[key] = np.array(result_mat[key])

    return result_py, result_mat



def matlab_SurfStatT(slm, contrast):
    # slm: ouput of SurfStatLinMod
    # contrast: np.ones(1)
    contrast = matlab.double(contrast.tolist())
    return surfstat_eng.SurfStatT(slm, contrast)

def matlab_SurfStatP(results):
    # results: SurfStatT(slm, contrast)
    return surfstat_eng.SurfStatP(results)

def matlab_SurfStatEdg(asurf):
    # asurf: np.array (n x 3)
    mystruct = surfstat_eng.struct('tri', matlab.double(asurf.tolist()))
    edg = surfstat_eng.SurfStatEdg(mystruct)
    return np.array(edg)
