# This file contains all the matlab_Surfstat* wrapper functions. Some of
# them are not yet implemented

import matlab.engine
import matlab
import numpy as np
import sys

def matlab_init_surfstat():
    global surfstat_eng
    surfstat_eng = matlab.engine.start_matlab()
    addpath = surfstat_eng.addpath('matlab')

# ==> SurfStatAvSurf.m <==
def matlab_SurfStatAvSurf(filenames, fun):
    sys.exit("Function matlab_SurfStatAvSurf is not implemented yet")

# ==> SurfStatAvVol.m <==
def matlab_SurfStatAvVol(filenames, fun, Nan):
    sys.exit("Function matlab_SurfStatAvVol is not implemented yet")

# ==> SurfStatColLim.m <==
def matlab_SurfStatColLim(clim):
    sys.exit("Function matlab_SurfStatColLim is not implemented yet")

# ==> SurfStatColormap.m <==
def matlab_SurfStatColormap(map):
    sys.exit("Function matlab_SurfStatColormap is not implemented yet")

# ==> SurfStatCoord2Ind.m <==
def matlab_SurfStatCoord2Ind(coord, surf):
    sys.exit("Function matlab_SurfStatCoord2Ind is not implemented yet")

# ==> SurfStatDataCursor.m <==
def matlab_SurfStatDataCursor(empt,event_obj):
    sys.exit("Function matlab_SurfStatDataCursor is not implemented yet")

# ==> SurfStatDataCursorP.m <==
def matlab_SurfStatDataCursorP(empt,event_obj):
    sys.exit("Function matlab_SurfStatDataCursorP is not implemented yet")

# ==> SurfStatDataCursorQ.m <==
def matlab_SurfStatDataCursorQ(empt,event_obj):
    sys.exit("Function matlab_SurfStatDataCursorQ is not implemented yet")

# ==> SurfStatDelete.m <==
def matlab_SurfStatDelete(varargin):
    sys.exit("Function matlab_SurfStatDelete is not implemented yet")

# ==> SurfStatEdg.m <==
def matlab_SurfStatEdg(asurf):
    mystruct = surfstat_eng.struct('tri', matlab.double(asurf.tolist()))
    edg = surfstat_eng.SurfStatEdg(mystruct)
    return np.array(edg)

# ==> SurfStatF.m <==
def matlab_SurfStatF(slm1, slm2):
    sys.exit("Function matlab_SurfStatF is not implemented yet")

# ==> SurfStatInd2Coord.m <==
def matlab_SurfStatInd2Coord(ind, surf):
    sys.exit("Function matlab_SurfStatInd2Coord is not implemented yet")

# ==> SurfStatInflate.m <==
def matlab_SurfStatInflate(surf, w, spherefile):
    sys.exit("Function matlab_SurfStatInflate is not implemented yet")

# ==> SurfStatLinMod.m <==
def matlab_SurfStatLinMod(T, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):

    # TODO implement ignored arguments

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

# ==> SurfStatListDir.m <==
def matlab_SurfStatListDir(d, exclude):
    sys.exit("Function matlab_SurfStatListDir is not implemented yet")

# ==> SurfStatMaskCut.m <==
def matlab_SurfStatMaskCut(surf):
    sys.exit("Function matlab_SurfStatMaskCut is not implemented yet")

# ==> SurfStatNorm.m <==
def matlab_SurfStatNorm(Y, mask=None, subdiv='s'):
	# Normalizes by subtracting the global mean, or dividing it. 
    # Inputs     	
    # Y      = numpy array of shape (n x v) or (n x v x k). 
    #          v=#vertices.
    # mask   = numpy boolean array of shape (1 x v). 
    #          True=inside the mask, False=outside. 
    # subdiv = 's' for Y=Y-Yav or 'd' for Y=Y/Yav.
    # Outputs
    # Y      = normalized data, numpy array of shape (n x v) or (n x v x k)
    # Yav    = mean of input Y along the mask, numpy array of shape (n x 1) or (n x k)   

    Y = matlab.double(Y.tolist())

    if mask is None and subdiv=='s':
        Y, Ya = surfstat_eng.SurfStatNorm(Y, nargout=2)
    
    elif mask is not None and subdiv=='s':
        mymask = np.array(mask, dtype=int)
        mymask = matlab.logical(matlab.double(mymask.tolist()))
        Y, Ya = surfstat_eng.SurfStatNorm(Y, mymask, nargout=2)

    elif mask is not None and subdiv=='d':
        mymask = np.array(mask, dtype=int)
        mymask = matlab.logical(matlab.double(mymask.tolist()))
        Y, Ya = surfstat_eng.SurfStatNorm(Y, mymask, subdiv, nargout=2)

    return np.array(Y), np.array(Ya)


# ==> SurfStatP.m <==
# TODO original matlab signature was SurfStatP(slm, mask, clusthresh):
def matlab_SurfStatP(results):
    return surfstat_eng.SurfStatP(results)

# ==> SurfStatPCA.m <==
def matlab_SurfStatPCA(Y, mask, X, k):
    sys.exit("Function matlab_SurfStatPCA is not implemented yet")

# ==> SurfStatPeakClus.m <==
def matlab_SurfStatPeakClus(slm, mask, thresh, reselspvert, edg):
    sys.exit("Function matlab_SurfStatPeakClus is not implemented yet")

# ==> SurfStatPlot.m <==
def matlab_SurfStatPlot(x, y, M, g, varargin):
    sys.exit("Function matlab_SurfStatPlot is not implemented yet")

# ==> SurfStatQ.m <==
def matlab_SurfStatQ(slm, mask):
    sys.exit("Function matlab_SurfStatQ is not implemented yet")

# ==> SurfStatROI.m <==
def matlab_SurfStatROI(centre, radius, surf):
    sys.exit("Function matlab_SurfStatROI is not implemented yet")

# ==> SurfStatROILabel.m <==
def matlab_SurfStatROILabel(lhlabel, rhlabel, nl, nr):
    sys.exit("Function matlab_SurfStatROILabel is not implemented yet")

# ==> SurfStatReadData.m <==
def matlab_SurfStatReadData(filenames, dirname, maxmem):
    sys.exit("Function matlab_SurfStatReadData is not implemented yet")

# ==> SurfStatReadData1.m <==
def matlab_SurfStatReadData1(filename):
    sys.exit("Function matlab_SurfStatReadData1 is not implemented yet")

# ==> SurfStatReadSurf.m <==
def matlab_SurfStatReadSurf(filenames,ab,numfields,dirname,maxmem):
    sys.exit("Function matlab_SurfStatReadSurf is not implemented yet")

# ==> SurfStatReadSurf1.m <==
def matlab_SurfStatReadSurf1(filename, ab, numfields):
    sys.exit("Function matlab_SurfStatReadSurf1 is not implemented yet")

# ==> SurfStatReadVol.m <==
def matlab_SurfStatReadVol(filenames,mask,step,dirname,maxmem):
    sys.exit("Function matlab_SurfStatReadVol is not implemented yet")

# ==> SurfStatReadVol1.m <==
def matlab_SurfStatReadVol1(file, Z, T):
    sys.exit("Function matlab_SurfStatReadVol1 is not implemented yet")

# ==> SurfStatResels.m <==
def matlab_SurfStatResels(slm, mask):
    sys.exit("Function matlab_SurfStatResels is not implemented yet")

# ==> SurfStatSmooth.m <==
def matlab_SurfStatSmooth(Y, surf, FWHM):
    sys.exit("Function matlab_SurfStatSmooth is not implemented yet")

# ==> SurfStatStand.m <==
def matlab_SurfStatStand(Y, mask, subtractordivide):
    sys.exit("Function matlab_SurfStatStand is not implemented yet")

# ==> SurfStatSurf2Vol.m <==
def matlab_SurfStatSurf2Vol(s, surf, template):
    sys.exit("Function matlab_SurfStatSurf2Vol is not implemented yet")
	
# ==> SurfStatT.m <==
def matlab_SurfStatT(slm, contrast):

    for key in slm.keys():
        if np.ndim(slm[key]) == 0:
            
            slm[key] = surfstat_eng.double(slm[key].item())
        else:
            slm[key] = matlab.double(slm[key].tolist())

    contrast = matlab.double(contrast.tolist())

    return surfstat_eng.SurfStatT(slm, contrast)
    
# ==> SurfStatView.m <==
def matlab_SurfStatView(struct, surf, title, background):
    sys.exit("Function matlab_SurfStatView is not implemented yet")

# ==> SurfStatView1.m <==
def matlab_SurfStatView1(struct, surf, varargin):
    sys.exit("Function matlab_SurfStatView1 is not implemented yet")

# ==> SurfStatViewData.m <==
def matlab_SurfStatViewData(data, surf, title, background):
    sys.exit("Function matlab_SurfStatViewData is not implemented yet")

# ==> SurfStatViews.m <==
def matlab_SurfStatViews(data, vol, z, layout):
    sys.exit("Function matlab_SurfStatViews is not implemented yet")

# ==> SurfStatVol2Surf.m <==
def matlab_SurfStatVol2Surf(vol, surf):
    sys.exit("Function matlab_SurfStatVol2Surf is not implemented yet")

# ==> SurfStatWriteData.m <==
def matlab_SurfStatWriteData(filename, data, ab):
    sys.exit("Function matlab_SurfStatWriteData is not implemented yet")

# ==> SurfStatWriteSurf.m <==
def matlab_SurfStatWriteSurf(filenames, surf, ab):
    sys.exit("Function matlab_SurfStatWriteSurf is not implemented yet")

# ==> SurfStatWriteSurf1.m <==
def matlab_SurfStatWriteSurf1(filename, surf, ab):
    sys.exit("Function matlab_SurfStatWriteSurf1 is not implemented yet")

# ==> SurfStatWriteVol.m <==
def matlab_SurfStatWriteVol(filenames, data, vol):
    sys.exit("Function matlab_SurfStatWriteVol is not implemented yet")

# ==> SurfStatWriteVol1.m <==
def matlab_SurfStatWriteVol(d, Z, T):
    sys.exit("Function matlab_SurfStatWriteVol is not implemented yet")
