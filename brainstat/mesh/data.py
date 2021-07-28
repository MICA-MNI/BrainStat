"""Operations on data on a mesh."""
from typing import Union

import numpy as np
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from .utils import mesh_edges


def mesh_smooth(
    Y: np.ndarray, surf: Union[dict, BSPolyData], FWHM: float
) -> np.ndarray:
    """Smooths surface data by repeatedly averaging over edges.

    Parameters
    ----------
    Y : numpy.ndarray
        Surface data of shape (n,v) or (n,v,k). v is the number of vertices,
        n is the number of observations, k is the number of variates.
    surf : dict, BSPolyData
        A dictionary with key 'tri' or 'lat', or a BSPolyData object of the
        surface. surf['tri'] is a numpy.ndarray  of shape (t,3), t is the
        triangle indices, or surf['lat'] is a numpy.ndarray of shape (nx,ny,nz),
        where (nx,ny,nz) is the volume size, values are 1=in, 0=out.
    FWHM : float
       Gaussian smoothing filter in mesh units.

    Returns
    -------
    numpy.ndarray
        Smoothed surface data of shape (n,v) or (n,v,k).
    """

    niter = int(np.ceil(pow(FWHM, 2) / (2 * np.log(2))))
    if isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype="float")
        if np.ndim(Y) == 2:
            n, v = np.shape(Y)
            k = 1
            isnum = True

        elif np.ndim(Y) == 3:
            n, v, k = np.shape(Y)
            isnum = True

    edg = mesh_edges(surf) + 1
    agg_1 = np.bincount(edg[:, 0], minlength=(v + 1)) * 2
    agg_2 = np.bincount(edg[:, 1], minlength=(v + 1)) * 2
    Y1 = (agg_1 + agg_2)[1:]

    if n > 1:
        print(" %i x %i surfaces to smooth, %% remaining: 100 " % (n, k))

    n10 = np.floor(n / 10)

    for i in range(0, n):

        if n10 != 0 and np.remainder(i + 1, n10) == 0:
            print("%s " % str(int(100 - (i + 1) / n10 * 10)), end="")

        for j in range(0, k):
            if isnum:
                if np.ndim(Y) == 2:
                    Ys = Y[i, :]

                elif np.ndim(Y) == 3:
                    Ys = Y[i, :, j]

                for itera in range(1, niter + 1):
                    Yedg = Ys[edg[:, 0] - 1] + Ys[edg[:, 1] - 1]
                    agg_tmp1 = np.bincount(edg[:, 0], Yedg, (v + 1))[1:]
                    agg_tmp2 = np.bincount(edg[:, 1], Yedg, (v + 1))[1:]
                    with np.errstate(invalid="ignore"):
                        Ys = (agg_tmp1 + agg_tmp2) / Y1

                if np.ndim(Y) == 2:
                    Y[i, :] = Ys

                elif np.ndim(Y) == 3:
                    Y[i, :, j] = Ys
    if n > 1:
        print("Done")

    return Y
