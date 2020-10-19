# This is the major file of the upcoming BrainStat/Surfstat port for python.
#
# All functions are prefixed with BrainStat. Each of them currently either
# redirects to a py_SurfStat* or matlab_Surfstat* implementation. The goals
# is to remove all the matlab_Surfstat* calls.

import sys
sys.path.append("python")

from SurfStatAvSurf import *
from SurfStatAvVol import *
from SurfStatEdg import *
from SurfStatF import *
from SurfStatLinMod import *
from SurfStatNorm import *
from SurfStatP import *
from SurfStatPeakClus import *
from SurfStatQ import *
from SurfStatResels import *
from SurfStatSmooth import *
from SurfStatStand import *
from SurfStatT import *


def BrainStatAvSurf(filenames, fun=np.add, output_surfstat=False):
    return py_SurfStatAvSurf(filenames, fun, output_surfstat)


def BrainStatAvVol(filenames, fun, Nan):
    return py_SurfStatAvVol(filenames, fun, Nan)


def BrainStatEdge(surf):
    return py_SurfStatEdge(surf)


def BrainStatF(slm1, slm2):
    return py_SurfStatF(slm1, slm2)


def BrainStatLinMod(Y, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):
    return py_SurfStatLinMod(Y, M, surf, niter, thetalim, drlim)


def BrainStatNorm(Y, mask=None, subdiv='s'):
    return py_SurfStatNorm(Y, mask, subdiv)


def BrainStatP(slm, mask=None, clusthresh=0.001):
    return py_SurfStatP(slm, mask, clusthresh)


def BrainStatPeakClus(slm, mask, thresh, reselspvert=None, edg=None):
    return py_SurfStatPeakClus(slm, mask, thresh, reselspvert, edg)


def BrainStatQ(slm, mask=None):
    return py_SurfStatQ(slm, mask)


def BrainStatResels(slm, mask=None):
    return py_SurfStatResels(slm, mask)


def BrainStatSmooth(Y, surf, FWHM):
    return py_SurfStatSmooth(Y, surf, FWHM)


def BrainStatStand(Y, mask=None, subtractordivide='s'):
    return py_SurfStatStand(Y, mask, subtractordivide)


def BrainStatT(slm, contrast):
    return py_SurfStatT(slm, contrast)
