import os
import numpy as np
import matlab.engine
import matlab

def init_surfstat():
    global surfstat_eng	
    surfstat_eng = matlab.engine.start_matlab()
    addpath = surfstat_eng.addpath('surfstat')

def SurfStatLinMod(T, M):
    # T, M: float, float
    T = surfstat_eng.double(T)
    M = surfstat_eng.double(M)
    return surfstat_eng.SurfStatLinMod(T, M)

def SurfStatT(slm, contrast):
    # slm: ouput of SurfStatLinMod
    # contrast: np.ones(1) 
    contrast = matlab.double(contrast.tolist())
    return surfstat_eng.SurfStatT(slm, contrast)

def SurfStatP(results):
    # results: SurfStatT(slm, contrast)
    return surfstat_eng.SurfStatP(results)

