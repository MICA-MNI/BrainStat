# This is the major file of the upcoming BrainStat/Surfstat port for python.
#
# All functions are prefixed with BrainStat. Each of them currently either
# redirects to a py_SurfStat* or matlab_Surfstat* implementation. The goals
# is to remove all the matlab_Surfstat* calls.

import sys
sys.path.append("python")

from brainstat.stats import *
from brainstat.stats import *
from SurfStatColLim import *
from SurfStatColormap import *
from SurfStatCoord2Ind import *
from SurfStatDataCursor import *
from SurfStatDataCursorP import *
from SurfStatDataCursorQ import *
from SurfStatDelete import *
from brainstat.stats import *
from brainstat.stats import *
from SurfStatInd2Coord import *
from SurfStatInflate import *
from brainstat.stats import *
from SurfStatListDir import *
from SurfStatMaskCut import *
from brainstat.stats import *
from brainstat.stats import *
from PCA import *
from brainstat.stats import *
from Plot import *
from brainstat.stats import *
from SurfStatROI import *
from SurfStatROILabel import *
from SurfStatReadData import *
from SurfStatReadData1 import *
from SurfStatReadSurf import *
from SurfStatReadSurf1 import *
from SurfStatReadVol import *
from SurfStatReadVol1 import *
from brainstat.stats import *
from brainstat.stats import *
from brainstat.stats import *
from SurfStatSurf2Vol import *
from brainstat.stats import *
from SurfStatView import *
from SurfStatView1 import *
from SurfStatViewData import *
from SurfStatViews import *
from SurfStatVol2Surf import *
from SurfStatWriteData import *
from SurfStatWriteSurf import *
from SurfStatWriteSurf1 import *
from SurfStatWriteVol import *
from SurfStatWriteVol1 import *


def BrainStatAvSurf(filenames, fun):
    return matlab_AvSurf(filenames, fun)


def BrainStatAvVol(filenames, fun, Nan):
    return matlab_AvVol(filenames, fun, Nan)


def BrainStatColLim(clim):
    return matlab_SurfStatColLim(clim)


def BrainStatColormap(map):
    return matlab_SurfStatColormap(map)


def BrainStatCoord2Ind(coord, surf):
    return matlab_SurfStatCoord2Ind(coord, surf)


def BrainStatDataCursor(empt,event_obj):
    return matlab_SurfStatDataCursor(empt,event_obj)


def BrainStatDataCursorP(empt,event_obj):
    return matlab_SurfStatDataCursorP(empt,event_obj)


def BrainStatDataCursorQ(empt,event_obj):
    return matlab_SurfStatDataCursorQ(empt,event_obj)


def BrainStatDelete(varargin):
    return matlab_SurfStatDelete(varargin)


def BrainStatEdge(surf):
    return matlab_Edge(surf)


def BrainStatF(slm1, slm2):
    return matlab_F(slm1, slm2)


def BrainStatInd2Coord(ind, surf):
    return matlab_SurfStatInd2Coord(ind, surf)


def BrainStatInflate(surf, w, spherefile):
    return matlab_SurfStatInflate(surf, w, spherefile)


def BrainStatLinMod(Y, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):
    return LinMod(Y, M, surf, niter, thetalim, drlim)


def BrainStatListDir(d, exclude):
    return matlab_SurfStatListDir(d, exclude)


def BrainStatMaskCut(surf):
    return matlab_SurfStatMaskCut(surf)


def BrainStatNorm(Y, mask, subdiv):
    return matlab_Norm(Y, mask, subdiv)


def BrainStatP(slm, mask, clusthresh):
    return matlab_P(slm, mask, clusthresh)


def BrainStatPCA(Y, mask, X, k):
    return matlab_PCA(Y, mask, X, k)


def BrainStatPeakClus(slm, mask, thresh, reselspvert, edg):
    return matlab_PeakClus(slm, mask, thresh, reselspvert, edg)


def BrainStatPlot(x, y, M, g, varargin):
    return matlab_Plot(x, y, M, g, varargin)


def BrainStatQ(slm, mask):
    return matlab_Q(slm, mask)


def BrainStatROI(centre, radius, surf):
    return matlab_SurfStatROI(centre, radius, surf)


def BrainStatROILabel(lhlabel, rhlabel, nl, nr):
    return matlab_SurfStatROILabel(lhlabel, rhlabel, nl, nr)


def BrainStatReadData(filenames, dirname, maxmem):
    return matlab_SurfStatReadData(filenames, dirname, maxmem)


def BrainStatReadData1(filename):
    return matlab_SurfStatReadData1(filename)


def BrainStatReadSurf(filenames,ab,numfields,dirname,maxmem):
    return matlab_SurfStatReadSurf(filenames,ab,numfields,dirname,maxmem)


def BrainStatReadSurf1(filename, ab, numfields):
    return matlab_SurfStatReadSurf1(filename, ab, numfields)


def BrainStatReadVol(filenames,mask,step,dirname,maxmem):
    return matlab_SurfStatReadVol(filenames,mask,step,dirname,maxmem)


def BrainStatReadVol1(file, Z, T):
    return matlab_SurfStatReadVol1(file, Z, T)


def BrainStatResels(slm, mask):
    return matlab_Resels(slm, mask)


def BrainStatSmooth(Y, surf, FWHM):
    return matlab_Smooth(Y, surf, FWHM)


def BrainStatStand(Y, mask, subtractordivide):
    return matlab_Stand(Y, mask, subtractordivide)


def BrainStatSurf2Vol(s, surf, template):
    return matlab_SurfStatSurf2Vol(s, surf, template)


def BrainStatT(slm, contrast):
    return matlab_T(slm, contrast)


def BrainStatView(struct, surf, title, background):
    return matlab_SurfStatView(struct, surf, title, background)


def BrainStatView1(struct, surf, varargin):
    return matlab_SurfStatView1(struct, surf, varargin)


def BrainStatViewData(data, surf, title, background):
    return matlab_SurfStatViewData(data, surf, title, background)


def BrainStatViews(data, vol, z, layout):
    return matlab_SurfStatViews(data, vol, z, layout)


def BrainStatVol2Surf(vol, surf):
    return matlab_SurfStatVol2Surf(vol, surf)


def BrainStatWriteData(filename, data, ab):
    return matlab_SurfStatWriteData(filename, data, ab)


def BrainStatWriteSurf(filenames, surf, ab):
    return matlab_SurfStatWriteSurf(filenames, surf, ab)


def BrainStatWriteSurf1(filename, surf, ab):
    return matlab_SurfStatWriteSurf1(filename, surf, ab)


def BrainStatWriteVol(filenames, data, vol):
    return matlab_SurfStatWriteVol(filenames, data, vol)


def BrainStatWriteVol(d, Z, T):
    return matlab_SurfStatWriteVol(d, Z, T)
