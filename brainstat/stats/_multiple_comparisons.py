# type: ignore
"""Multiple comparison corrections."""
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import concatenate as cat
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix
from scipy.special import betaln, gamma, gammaln

from brainstat._typing import ArrayLike
from brainstat.mesh.utils import mesh_edges
from brainstat.stats.utils import colon, interp1, ismember, row_ismember


def _fdr(self) -> np.ndarray:
    """Q-values for False Discovey Rate of resels.

    Parameters
    ----------
    self : brainstat.stats.SLM.SLM
        SLM object with computed t-values.

    Returns
    -------
    numpy.array
        Q-values for false discovery rate of resels.

    """
    l, v = np.shape(self.t)

    if self.mask is None:
        self.mask = np.ones((v), dtype="bool")

    df = np.zeros((2, 2))
    ndf = len(np.array([self.df]))
    df[0, 0:ndf] = self.df
    df[1, 0:2] = np.array([self.df])[ndf - 1]

    if self.dfs is not None:
        df[0, ndf - 1] = self.dfs[0, self.mask > 0].mean()

    if self.du is not None:
        resels, reselspvert, edg = compute_resels(self)
    else:
        reselspvert = np.ones((v))

    varA = np.append(10, self.t[0, self.mask.astype(bool)])
    P_val = stat_threshold(df=df, p_val_peak=varA, nvar=self.k, nprint=0)[0]
    P_val = P_val[1 : len(P_val)]
    nx = len(P_val)
    index = P_val.argsort()
    P_sort = P_val[index]
    r_sort = reselspvert[index]
    c_sort = np.cumsum(r_sort)
    P_sort = P_sort / (c_sort + (c_sort <= 0)) * (c_sort > 0) * r_sort.sum()
    m = 1
    Q_sort = np.zeros((1, nx))

    for i in np.arange(nx, 0, -1):
        if P_sort[i - 1] < m:
            m = P_sort[i - 1]
        Q_sort[0, i - 1] = m

    Q_tmp = np.zeros((1, nx))
    Q_tmp[0, index] = Q_sort

    Q = np.ones((self.mask.shape[0]))
    Q[self.mask] = np.squeeze(Q_tmp[0, :])

    return Q


valid_rft_output = Union[
    Tuple[dict, dict, dict, np.ndarray], Tuple[dict, List, List, List]
]


def _random_field_theory(self) -> valid_rft_output:
    """Corrected P-values for vertices and clusters.
    Parameters
    ----------
    self : brainstat.stats.SLM.SLM
        SLM object with computed t-values.

    Returns
    -------
    pval : a dictionary with keys 'P', 'C', 'mask'.
        pval['P'] : 2D numpy array of shape (1,v).
            Corrected P-values for vertices.
        pval['C'] : 2D numpy array of shape (1,v).
            Corrected P-values for clusters.
    peak : a dictionary with keys 't', 'vertid', 'clusid', 'P'.
        peak['t'] : 2D numpy array of shape (np,1).
            Peaks (local maxima).
        peak['vertid'] : 2D numpy array of shape (np,1).
            Vertex.
        peak['clusid'] : 2D numpy array of shape (np,1).
            Cluster id numbers.
        peak['P'] : 2D numpy array of shape (np,1).
            Corrected P-values for the peak.
    clus : a dictionary with keys 'clusid', 'nverts', 'resels', 'P.'
        clus['clusid'] : 2D numpy array of shape (nc,1).
            Cluster id numbers
        clus['nverts'] : 2D numpy array of shape (nc,1).
            Number of vertices in cluster.
        clus['resels'] : 2D numpy array of shape (nc,1).
            resels in the cluster.
        clus['P'] : 2D numpy array of shape (nc,1).
            Corrected P-values for the cluster.
    clusid : 2D numpy array of shape (1,v).
        Cluster id's for each vertex.
    Reference: Worsley, K.J., Andermann, M., Koulis, T., MacDonald, D.
    & Evans, A.C. (1999). Detecting changes in nonisotropic images.
    Human Brain Mapping, 8:98-101.
    """
    l, v = np.shape(self.t)

    if self.mask is None:
        self.mask = np.ones((v), dtype=bool)

    df = np.zeros((2, 2))
    ndf = len(np.array([self.df]))
    df[0, 0:ndf] = self.df
    df[1, 0:2] = np.array([self.df])[ndf - 1]

    if self.dfs is not None:
        df[0, ndf - 1] = self.dfs[0, self.mask > 0].mean()

    if v == 1:
        varA = varA = np.concatenate((np.array([10]), self.t[0]))
        pval = {}
        pval["P"] = stat_threshold(df=df, p_val_peak=varA, nvar=self.k, nprint=0)[0]
        pval["P"] = pval["P"][1]
        peak = []
        clus = []
        clusid = []
        # only a single p-value is returned, and function is stopped.
        return pval, peak, clus, clusid

    if self.cluster_threshold < 1:
        thresh_tmp = stat_threshold(
            df=df, p_val_peak=self.cluster_threshold, nvar=self.k, nprint=0
        )[0]
        thresh = float(thresh_tmp[0])
    else:
        thresh = self.cluster_threshold

    resels, reselspvert, edg = compute_resels(self)
    N = self.mask.sum()

    if np.max(self.t[0, self.mask]) < thresh:
        pval = {}
        varA = np.concatenate((np.array([[10]]), self.t), axis=1)
        pval["P"] = stat_threshold(
            search_volume=resels,
            num_voxels=N,
            fwhm=1,
            df=df,
            p_val_peak=varA.flatten(),
            nvar=self.k,
            nprint=0,
        )[0]
        pval["P"] = pval["P"][1 : v + 1]
        pval["C"] = None
        peak = {"t": None, "clusid": None, "vertid": None, "P": None}
        clus = {"clusid": None, "nverts": None, "resels": None, "P": None}
        clusid = None
    else:
        peak, clus, clusid = peak_clus(self, thresh, reselspvert, edg)
        self.t = self.t.reshape(1, self.t.size)
        varA = np.concatenate((np.array([[10]]), peak["t"].T, self.t), axis=1)
        varB = np.concatenate((np.array([[10]]), clus["resels"]))
        pp, clpval, _, _, _, _, = stat_threshold(
            search_volume=resels,
            num_voxels=N,
            fwhm=1,
            df=df,
            p_val_peak=varA.flatten(),
            cluster_threshold=thresh,
            p_val_extent=varB,
            nvar=float(self.k),
            nprint=0,
        )
        lenPP = len(pp[1 : len(peak["t"]) + 1])
        peak["P"] = pp[1 : len(peak["t"]) + 1].reshape(lenPP, 1)
        pval = {}
        pval["P"] = pp[len(peak["t"]) + np.arange(1, v + 1)]

        if self.k > 1:
            j = np.arange(self.k)[::-2]
            sphere = np.zeros((1, int(self.k)))
            sphere[:, j] = np.exp(
                (j + 1) * np.log(2)
                + (j / 2) * np.log(math.pi)
                + gammaln((self.k + 1) / 2)
                - gammaln(j + 1)
                - gammaln((self.k + 1 - j) / 2)
            )
            sphere = sphere * np.power(4 * np.log(2), -np.arange(0, self.k) / 2) / ndf
            varA = np.convolve(resels.flatten(), sphere.flatten())
            varB = np.concatenate((np.array([[10]]), clus["resels"]))
            pp, clpval, _, _, _, _, = stat_threshold(
                search_volume=varA,
                num_voxels=math.inf,
                fwhm=1.0,
                df=df,
                cluster_threshold=thresh,
                p_val_extent=varB,
                nprint=0,
            )

        clus["P"] = clpval[1 : len(clpval)]
        x = np.concatenate((np.array([[0]]), clus["clusid"]), axis=0)
        y = np.concatenate((np.array([[1]]), clus["P"]), axis=0)
        pval["C"] = interp1d(x.flatten(), y.flatten())(clusid)

    tlim = stat_threshold(
        search_volume=resels,
        num_voxels=N,
        fwhm=1,
        df=df,
        p_val_peak=np.array([0.5, 1]),
        nvar=float(self.k),
        nprint=0,
    )[0]
    tlim = tlim[1]
    pval["P"] = pval["P"] * (self.t[0, :] > tlim) + (self.t[0, :] <= tlim)

    return pval, peak, clus, clusid


def stat_threshold(
    search_volume: Union[float, ArrayLike] = 0,
    num_voxels: Union[float, ArrayLike] = 1,
    fwhm: float = 0.0,
    df: Union[float, ArrayLike] = math.inf,
    p_val_peak: float = 0.05,
    cluster_threshold: float = 0.001,
    p_val_extent: Union[float, ArrayLike] = 0.05,
    nconj: float = 1,
    nvar: int = 1,
    EC_file: Optional[bool] = None,
    nprint: int = 5,
) -> Tuple[np.ndarray, ...]:
    """Thresholds and P-values of peaks and clusters of random fields in any D.

    Parameters
    ----------
    search_volume : a float, or a list, or a numpy array
        volume of the search region in mm^3.
    num_voxels : a float, or int, or list, or 1D numpy array
        number of voxels (3D) or pixels (2D) in the search volume.
    fwhm : a float, or int.
        fwhm in mm of a smoothing kernel applied to the data.
    df : a float, or int, or list, or  array of shape (2,2)
        degrees of freedom.
    p_val_peak : a float, or 1D array of shape (y,)
        desired P-values for peaks.
    cluster_threshold: a float
        scalar threshold of the image for clusters
    p_val_extent : a float, or list, or 1D array of shape (y,)
        desired P-values for spatial extents of clusters of contiguous
        voxels above the cluster_threshold
    nconj : a float, or int
        number of conjunctions
    nvar :an int, list or 1D array of 1 or 2 integers
        number of variables for multivariate equivalents of T and F
        statistics

    Returns
    -------
    peak_threshold :
        thresholds for local maxima or peaks
    extent_threshold :

    peak_threshold_1
        height of a single peak chosen in advance
    extent_threshold_1
        extent of a single cluster chosen in advance
    t :

    rho :
    """

    def gammalni(n):
        x = math.inf * np.ones(n.shape)
        x[n >= 0] = gammaln(n[n >= 0])
        return x

    def minterp1(x, y, ix):
        # interpolates only the monotonically increasing values of x at ix
        n = x.size
        ix = np.array(ix)
        ix_shape = ix.shape
        ix = ix.flatten("F")

        mx = np.array(x[0], ndmin=1)
        my = np.array(y[0], ndmin=1)
        xx = x[0]
        for i in range(0, n):
            if x[i] > xx:
                xx = x[i]
                mx = np.append(mx, xx)
                my = np.append(my, y[i])

        out = []
        for i in range(0, ix.size):
            if ix[i] < mx[0] or ix[i] > mx[-1]:
                out.append(math.nan)
            else:
                out.append(interp1(mx, my, ix[i]))
        out = np.reshape(out, ix_shape, order="F")
        return out

    # Deal with the input

    # Make sure all input is in np.array format.
    fwhm = np.array(fwhm, ndmin=1)
    search_volume = np.array(search_volume, ndmin=2)
    num_voxels = np.array(num_voxels)
    df = np.array(df, ndmin=2)
    nvar = np.array(nvar, dtype=int)
    p_val_peak = np.array(p_val_peak, ndmin=1)
    p_val_extent = np.array(p_val_extent, ndmin=1)

    # Set the FWHM
    if fwhm.ndim == 1:
        fwhm = np.expand_dims(fwhm, axis=0)
        fwhm = np.r_[fwhm, fwhm]
    if np.shape(fwhm)[1] == 1:
        scale = 1
    else:
        scale = fwhm[0, 1] / fwhm[0, 0]
        fwhm = fwhm[:, 0]
    isscale = scale > 1

    # Set the number of voxels
    if num_voxels.size == 1:
        num_voxels = np.append(num_voxels, 1)

    # Set the search volume.
    if search_volume.shape[1] == 1:
        radius = (search_volume / (4 / 3 * math.pi)) ** (1 / 3)
        search_volume = np.c_[
            np.ones(radius.shape), 4 * radius, 2 * radius ** 2 * math.pi, search_volume
        ]

    if search_volume.shape[0] == 1:
        search_volume = np.concatenate(
            (
                search_volume,
                np.concatenate(
                    (np.ones((1, 1)), np.zeros((1, search_volume.size - 1))), axis=1
                ),
            ),
            axis=0,
        )

    lsv = search_volume.shape[1]
    if all(fwhm > 0):
        fwhm_inv = all(fwhm > 0) / fwhm + any(fwhm <= 0)
    else:
        fwhm_inv = np.zeros(fwhm.shape)
    if fwhm_inv.ndim == 1:
        fwhm_inv = np.expand_dims(fwhm_inv, axis=1)

    resels = search_volume * fwhm_inv ** np.arange(0, lsv)
    invol = resels * (4 * np.log(2)) ** (np.arange(0, lsv) / 2)

    D = []
    for k in range(2):
        D.append(np.max(np.argwhere(invol[k, :])))

    # determines which method was used to estimate fwhm (see fmrilm or multistat):
    df_limit = 4

    if df.size == 1:
        df = np.c_[df, np.zeros((1, 1))]
    if df.shape[0] == 1:
        infs = np.array([math.inf, math.inf], ndmin=2)
        df = np.r_[df, infs, infs]
    if df.shape[1] == 1:
        df = np.c_[df, df]
        df[0, 1] = 0
    if df.shape[0] == 2:
        df = np.r_[df, np.expand_dims(df[1, :], axis=0)]

    # is_tstat=1 if it is a t statistic
    is_tstat = df[0, 1] == 0
    if is_tstat:
        df1 = 1
        df2 = df[0, 0]
    else:
        df1 = df[0, 0]
        df2 = df[0, 1]
    if df2 >= 1000:
        df2 = math.inf
    df0 = df1 + df2

    dfw1 = df[1:3, 0].astype("float64")
    dfw2 = df[1:3, 1].astype("float64")
    dfw1[dfw1 >= 1000] = math.inf
    dfw2[dfw2 >= 1000] = math.inf
    if nvar.size == 1:
        nvar = np.r_[nvar, int(np.round(df1))]

    if isscale and (D[1] > 1 or nvar[0] > 1 | df2 < math.inf):
        print(D)
        print(nvar)
        print(df2)
        print("Cannot do scale space.")
        return
    Dlim = D + np.array([scale > 1, 0])
    DD = Dlim + nvar - 1

    # Values of the F statistic:
    t = (np.arange(1000, 0, -1) / 100) ** 4

    # Find the upper tail probs cumulating the F density using Simpson's rule:
    if math.isinf(df2):
        u = df1 * t
        b = (
            np.exp(-u / 2 - np.log(2 * math.pi) / 2 + np.log(u) / 4)
            * df1 ** (1 / 4)
            * 4
            / 100
        )
    else:
        u = df1 * t / df2
        b = (
            np.exp(
                -df0 / 2 * np.log(1 + u) + np.log(u) / 4 - betaln(1 / 2, (df0 - 1) / 2)
            )
            * (df1 / df2) ** (1 / 4)
            * 4
            / 100
        )

    t = np.r_[t, 0]
    b = np.r_[b, 0]
    n = t.size
    sb = np.cumsum(b)
    sb1 = np.cumsum(b * (-1) ** np.arange(1, n + 1))
    pt1 = sb + sb1 / 3 - b / 3
    pt2 = sb - sb1 / 3 - b / 3
    tau = np.zeros((n, DD[0] + 1, DD[1] + 1))
    tau[0:n:2, 0, 0] = pt1[0:n:2]
    tau[1:n:2, 0, 0] = pt2[1:n:2]
    tau[n - 1, 0, 0] = 1
    tau[tau > 1] = 1

    # Find the EC densities:
    u = df1 * t
    for d in range(1, np.max(DD) + 1):
        e_loop = np.min([np.min(DD), d]) + 1
        for e in range(0, e_loop):
            s1 = 0
            cons = -((d + e) / 2 + 1) * np.log(math.pi) + gammaln(d) + gammaln(e + 1)
            for k in colon(0, (d - 1 + e) / 2):
                j, i = np.meshgrid(np.arange(0, k + 1), np.arange(0, k + 1))
                if df2 == math.inf:
                    q1 = np.log(math.pi) / 2 - ((d + e - 1) / 2 + i + j) * np.log(2)
                else:
                    q1 = (
                        (df0 - 1 - d - e) * np.log(2)
                        + gammaln((df0 - d) / 2 + i)
                        + gammaln((df0 - e) / 2 + j)
                        - gammalni(df0 - d - e + i + j + k)
                        - ((d + e - 1) / 2 - k) * np.log(df2)
                    )
                q2 = (
                    cons
                    - gammalni(i + 1)
                    - gammalni(j + 1)
                    - gammalni(k - i - j + 1)
                    - gammalni(d - k - i + j)
                    - gammalni(e - k - j + i + 1)
                )
                s2 = np.sum(np.exp(q1 + q2))
                if s2 > 0:
                    s1 = s1 + (-1) ** k * u ** ((d + e - 1) / 2 - k) * s2

            if df2 == math.inf:
                s1 = s1 * np.exp(-u / 2)
            else:
                s1 = s1 * np.exp(-(df0 - 2) / 2 * np.log(1 + u / df2))

            if DD[0] >= DD[1]:
                tau[:, d, e] = s1
                if d <= np.min(DD):
                    tau[:, e, d] = s1
            else:
                tau[:, e, d] = s1
                if d <= np.min(DD):
                    tau[:, d, e] = s1

    # For multivariate statistics, add a sphere to the search region:
    a = np.zeros((2, np.max(nvar)))
    for k in range(0, 2):
        j = colon((nvar[k] - 1), 0, -2)
        a[k, j] = np.exp(
            j * np.log(2)
            + j / 2 * np.log(math.pi)
            + gammaln((nvar[k] + 1) / 2)
            - gammaln((nvar[k] + 1 - j) / 2)
            - gammaln(j + 1)
        )

    rho = np.zeros((n, Dlim[0] + 1, Dlim[1] + 1))

    for k in range(0, nvar[0]):
        for l in range(0, nvar[1]):
            rho = (
                rho
                + a[0, k] * a[1, l] * tau[:, k : Dlim[0] + k + 1, l : Dlim[1] + l + 1]
            )

    if is_tstat:
        if all(nvar == 1):
            t = np.r_[np.sqrt(t[0 : n - 1]), -np.sqrt(t)[::-1]]
            rho = np.r_[rho[0 : n - 1, :, :], rho[::-1, :, :]] / 2
            for i in range(0, D[0] + 1):
                for j in range(0, D[1] + 1):
                    rho[n - 1 + np.arange(0, n), i, j] = (
                        -((-1) ** (i + j)) * rho[n - 1 + np.arange(0, n), i, j]
                    )
            rho[n - 1 + np.arange(0, n), 0, 0] = rho[n - 1 + np.arange(0, n), 0, 0] + 1
            n = 2 * n - 1
        else:
            t = np.sqrt(t)

    # For scale space.
    if scale > 1:
        kappa = D[0] / 2
        tau = np.zeros(n, D[0] + 1)
        for d in range(0, D[0] + 1):
            s1 = 0
            for k in range(0, d / 2 + 1):
                s1 = (
                    s1 + (-1)
                    ^ k
                    / (1 - 2 * k)
                    * np.exp(
                        gammaln(d + 1)
                        - gammaln(k + 1)
                        - gammaln(d - 2 * k + 1)
                        + (1 / 2 - k) * np.log(kappa)
                        - k * np.log(4 * math.pi)
                    )
                    * rho[:, d + 1 - 2 * k, 1]
                )
            if d == 0:
                cons = np.log(scale)
            else:
                cons = (1 - 1 / scale ** d) / d
            tau[:, d] = rho[:, d, 1] * (1 + 1 / scale ** d) / 2 + s1 * cons
        rho[:, 0 : D[1], 0] = tau

    if D[1] == 0:
        d = D[0]
        if nconj > 1:
            # Conjunctions
            b = gamma((np.arange(1, d + 2) / 2)) / gamma(1 / 2)
            for i in range(0, d + 1):
                rho[:, i, 0] = rho[:, i, 0] / b[i]
            m1 = np.zeros((n, d + 1, d + 1))
            for i in range(0, d + 1):
                j = np.arange(i, d + 1)
                m1[:, i, j] = rho[:, j - i, 0]
            m2 = np.zeros(m1.shape)
            for _ in range(2, nconj + 1):
                for i in range(0, d + 1):
                    for j in range(0, d + 1):
                        m2[:, i, j] = np.sum(
                            rho[:, 0 : d - i + 1, 0] * m1[:, i : d + 1, j], axis=1
                        )
                m1 = m2
            for i in range(0, d + 1):
                rho[:, i, 0] = m1[:, 0, i] * b[i]
        if EC_file is not None:
            raise ValueError(
                "EC File support has not been implemented. We are not living in the 90s anymore."
            )

    if all(fwhm > 0):
        pval_rf = np.zeros(n)
        for i in range(0, D[0] + 1):
            for j in range(0, D[1] + 1):
                pval_rf = pval_rf + invol[0, i] * invol[1, j] * rho[:, i, j]
    else:
        pval_rf = math.inf

    # Bonferonni
    pt = rho[:, 0, 0]
    pval_bon = abs(np.prod(num_voxels)) * pt

    # Minimum of the two
    if type(pval_rf) != type(pval_bon):
        pval = np.minimum(pval_rf, pval_bon)
    else:
        if isinstance(pval_rf, float) and isinstance(pval_bon, float):
            pval = np.minimum(pval_rf, pval_bon)
        if isinstance(pval_rf, np.ndarray) and isinstance(pval_bon, np.ndarray):
            # this will ignore nan values in arrays while finding minimum
            pval_tmp = np.concatenate(
                (
                    pval_rf.reshape(pval_rf.shape[0], 1),
                    pval_bon.reshape(pval_bon.shape[0], 1),
                ),
                axis=1,
            )
            pval = np.nanmin(pval_tmp, axis=1)

    tlim = 1
    if p_val_peak.flatten()[0] <= tlim:
        peak_threshold = minterp1(pval, t, p_val_peak)
        if p_val_peak.size <= nprint:
            print(peak_threshold)
    else:
        # p_val_peak is treated as peak value
        P_val_peak = interp1(t, pval, p_val_peak)
        i = np.isnan(P_val_peak)
        P_val_peak[i] = is_tstat and (p_val_peak[i] < 0)
        peak_threshold = P_val_peak
        if p_val_peak.size <= nprint:
            print(P_val_peak)

    if np.all(fwhm <= 0) or np.any(num_voxels < 0):
        peak_threshold_1 = p_val_peak + float("nan")
        extent_threshold = p_val_extent + float("nan")
        extent_threshold_1 = extent_threshold + float("nan")
        return (
            peak_threshold,
            extent_threshold,
            peak_threshold_1,
            extent_threshold_1,
            t,
            rho,
        )

    # Cluster threshold:

    if cluster_threshold > tlim:
        tt = cluster_threshold
    else:
        # cluster_threshold is treated as a probability
        tt = minterp1(pt, t, cluster_threshold)

    d = np.sum(D)
    rhoD = interp1(t, rho[:, D[0], D[1]], tt)
    p = interp1(t, pt, tt)

    # Pre-selected peak
    pval = rho[:, D[0], D[1]] / rhoD
    if p_val_peak.flatten()[0] <= tlim:
        peak_threshold_1 = minterp1(pval, t, p_val_peak)
        if p_val_peak.size <= nprint:
            print(peak_threshold_1)
    else:
        # p_val_peak is treated as a peak value
        P_val_peak_1 = interp1(t, pval, p_val_peak)
        i = np.isnan(P_val_peak)
        P_val_peak_1[i] = is_tstat and (p_val_peak[i] < 0)
        peak_threshold_1 = P_val_peak_1
        if p_val_peak.size <= nprint:
            print(P_val_peak_1)

    if d == 0 or nconj > 1 or nvar[0] > 1 or scale > 1:
        extent_threshold = p_val_extent + float("nan")
        extent_threshold_1 = extent_threshold
        if p_val_extent.size <= nprint:
            print(extent_threshold)
            print(extent_threshold_1)
        return (
            peak_threshold,
            extent_threshold,
            peak_threshold_1,
            extent_threshold_1,
            t,
            rho,
        )

    # Expected number of clusters
    EL = invol[0, D[0]] * invol[1, D[1]] * rhoD
    cons = (
        gamma(d / 2 + 1)
        * (4 * np.log(2)) ** (d / 2)
        / fwhm[0] ** D[0]
        / fwhm[1] ** D[1]
        * rhoD
        / p
    )

    if df2 == math.inf and dfw1[0] == math.inf and dfw1[1] == math.inf:
        if p_val_extent.flatten()[0] <= tlim:
            pS = -np.log(1 - p_val_extent) / EL
            extent_threshold = (-np.log(pS)) ** (d / 2) / cons
            pS = -np.log(1 - p_val_extent)
            extent_threshold_1 = (-np.log(pS)) ** (d / 2) / cons
            if p_val_extent.size <= nprint:
                print(extent_threshold)
                print(extent_threshold_1)
        else:
            # p_val_extent is now treated as a spatial extent:
            pS = np.exp(-((p_val_extent * cons) ** (2 / d)))
            P_val_extent = 1 - np.exp(-pS * EL)
            extent_threshold = P_val_extent
            P_val_extent_1 = 1 - np.exp(-pS)
            extent_threshold_1 = P_val_extent_1
            if p_val_extent.size <= nprint:
                print(P_val_extent)
                print(P_val_extent_1)
    else:
        # Find dbn of S by taking logs then using fft for convolution:
        ny = 2 ** 12
        a = d / 2
        b2 = a * 10 * np.max([np.sqrt(2 / (np.min([df1 + df2, np.min(dfw1)]))), 1])
        if df2 < math.inf:
            b1 = a * np.log((1 - (1 - 0.000001) ** (2 / (df2 - d))) * df2 / 2)
        else:
            b1 = a * np.log(-np.log(1 - 0.000001))

        dy = (b2 - b1) / ny
        b1 = round(b1 / dy) * dy
        y = np.arange(0, ny) * dy + b1
        numrv = (
            1
            + (d + (D[0] > 0) + (D[1] > 0)) * (df2 < math.inf)
            + (D[0] * (dfw1[0] < math.inf) + (dfw2[0] < math.inf)) * (D[0] > 0)
            + (D[1] * (dfw1[1] < math.inf) + (dfw2[1] < math.inf)) * (D[1] > 0)
        )
        f = np.zeros((ny, numrv))
        if f.ndim == 1:
            f = np.expand_dims(f, axis=1)
        mu = np.zeros(numrv)
        if df2 < math.inf:
            # Density of log(Beta(1,(df2-d)/2)^(d/2)):
            yy = np.exp(y / a) / df2 * 2
            yy = yy * (yy < 1)
            f[:, 0] = (1 - yy) ** ((df2 - d) / 2 - 1) * ((df2 - d) / 2) * yy / a
            mu[0] = np.exp(
                gammaln(a + 1)
                + gammaln((df2 - d + 2) / 2)
                - gammaln((df2 + 2) / 2)
                + a * np.log(df2 / 2)
            )
        else:
            # Density of log(exp(1)^(d/2)):
            yy = np.exp(y / a)
            f[:, 0] = np.exp(-yy) * yy / a
            mu[0] = np.exp(gammaln(a + 1))

        nuv = np.array([])
        aav = np.array([])
        if df2 < math.inf:
            nuv = df2 + 2 - np.arange(1, d + 1)
            aav = np.ones((1, d)) * (-1 / 2)
            for k in range(0, 2):
                if D[k] > 0:
                    nuv = np.append(df1 + df2 - D[k], nuv)
                    aav = np.append(D[k] / 2, aav)

        for k in range(0, 2):
            if dfw1[k] < math.inf and D[k] > 0:
                if dfw1[k] > df_limit:
                    nuv = np.append(
                        nuv, dfw1[k] - dfw1[k] / dfw2[k] - np.arange(0, D[k])
                    )
                else:
                    nuv = np.append(
                        nuv, (dfw1[k] - dfw1[k] / dfw2[k]) * np.ones((1, D[k]))
                    )
                aav = np.append(aav, (1 / 2) * np.ones((1, D[k])))
            if dfw2[k] < math.inf:
                nuv = np.append(nuv, dfw2[k])
                aav = np.append(aav, -D[k] / 2)

        for i in range(0, numrv - 1):
            nu = nuv[i]
            aa = aav[i]
            yy = y / aa + np.log(nu)
            # Density of log((chi^2_nu/nu)^aa):
            f[:, i + 1] = np.exp(
                nu / 2 * yy - np.exp(yy) / 2 - (nu / 2) * np.log(2) - gammaln(nu / 2)
            ) / abs(aa)
            mu[i + 1] = np.exp(
                gammaln(nu / 2 + aa) - gammaln(nu / 2) - aa * np.log(nu / 2)
            )

        # Check: plot(y,f); sum(f*dy,1) should be 1

        omega = 2 * math.pi * np.arange(0, ny) / ny / dy
        shift = (np.cos(-b1 * omega) + np.sin(-b1 * omega) * 1j) * dy
        prodfft = np.prod(np.fft.fft(f, axis=0), axis=1) * shift ** (numrv - 1)

        # Density of Y=log(B^(d/2)*U^(d/2)/sqrt(det(Q)))
        ff = np.real(np.fft.ifft(prodfft, axis=0))
        # Check: plot(y,ff); sum(ff*dy) should be 1
        mu0 = np.prod(mu)
        # Check: plot(y,ff.*exp(y)); sum(ff.*exp(y)*dy.*(y<10)) should equal mu0
        alpha = (
            p
            / rhoD
            / mu0
            * fwhm[0] ** D[0]
            * fwhm[1] ** D[1]
            / (4 * np.log(2)) ** (d / 2)
        )

        # Integrate the density to get the p-value for one cluster:
        pS = np.cumsum(np.flip(ff)) * dy
        pS = np.flip(pS)

        # The number of clusters is Poisson with mean EL:
        pSmax = 1 - np.exp(-pS * EL)
        if p_val_extent.flatten()[0] <= tlim:
            yval = minterp1(-pSmax, y, -p_val_extent)
            # Spaytial extent is alpha*exp(Y) -dy/2 correction for mid-point rule:
            extent_threshold = alpha * np.exp(yval - dy / 2)
            # For a single cluster:
            yval = minterp1(-pS, y, -p_val_extent)
            extent_threshold_1 = alpha * np.exp(yval - dy / 2)
            if p_val_extent.size <= nprint:
                print(extent_threshold)
                print(extent_threshold_1)
        else:
            # p_val_extent is now treated as a spatial extent:
            logpval = np.log(p_val_extent / alpha + (p_val_extent <= 0)) + dy / 2
            P_val_extent = interp1(y, pSmax, logpval)
            extent_threshold = P_val_extent * (p_val_extent > 0) + (p_val_extent <= 0)
            # For a single cluster:
            P_val_extent_1 = interp1(y, pS, logpval)
            extent_threshold_1 = P_val_extent_1 * (p_val_extent > 0) + (
                p_val_extent <= 0
            )
            if p_val_extent.size <= nprint:
                print(P_val_extent)
                print(P_val_extent_1)

    return (
        peak_threshold,
        extent_threshold,
        peak_threshold_1,
        extent_threshold_1,
        t,
        rho,
    )


def peak_clus(
    self,
    thresh: float,
    reselspvert: Optional[np.ndarray] = None,
    edg: Optional[np.ndarray] = None,
) -> Tuple[dict, dict, np.ndarray]:
    """Finds peaks (local maxima) and clusters for surface data.
    Parameters
    ----------
    thresh : float,
        clusters are vertices where slm['t'][0,mask]>=thresh.
    reselspvert : numpy array of shape (v),
        resels per vertex, by default: np.ones(v).
    edg :  numpy array of shape (e,2), dtype=int,
        edge indices, by default computed from mesh_edges function.


    Returns
    -------
    peak : a dictionary with keys 't', 'vertid', 'clusid'.
        peak['t'] : numpy array of shape (np,1),
            array of peaks (local maxima).
        peak['vertid] : numpy array of shape (np,1),
            array of vertex id's (1-based).
        peak['clusid'] : numpy array of shape (np,1),
            array of cluster id's that contain the peak.
    clus : a dictionary with keys 'clusid', 'nverts', 'resels'.
        clus['clusid'] : numpy array of shape (nc,1),
            array of cluster id numbers.
        clus['nverts'] : numpy array of shape (nc,1),
            array of number of vertices in the cluster.
        clus['resels'] : numpy array of shape (nc,1),
            array of resels in the cluster.
    clusid : numpy array of shape (1,v),
        array of cluster id's for each vertex.
    """

    if self.k > 1 and self.df is None:
        raise ValueError("If k>1 then df must be defined.")

    if edg is None:
        edg = mesh_edges(self)

    l, v = np.shape(self.t)
    slm_t = copy.deepcopy(self.t)
    slm_t[0, ~self.mask.astype(bool)] = slm_t[0, :].min()
    t1 = slm_t[0, edg[:, 0]]
    t2 = slm_t[0, edg[:, 1]]
    islm = np.ones((1, v))
    islm[0, edg[t1 < t2, 0]] = 0
    islm[0, edg[t2 < t1, 1]] = 0
    lmvox = np.argwhere(islm)[:, 1] + 1
    excurset = np.array(slm_t[0, :] >= thresh, dtype=int)
    n = excurset.sum()

    if n < 1:
        peak = []
        clus = []
        clusid = []
        return peak, clus, clusid

    voxid = np.cumsum(excurset)
    edg = voxid[edg[np.all(excurset[edg], 1), :]]
    nf = np.arange(1, n + 1)

    # Find cluster id's in nf (from Numerical Recipes in C, page 346):
    for el in range(1, edg.shape[0] + 1):
        j = edg[el - 1, 0]
        k = edg[el - 1, 1]
        while nf[j - 1] != j:
            j = nf[j - 1]
        while nf[k - 1] != k:
            k = nf[k - 1]
        if j != k:
            nf[j - 1] = k

    for j in range(1, n + 1):
        while nf[j - 1] != nf[nf[j - 1] - 1]:
            nf[j - 1] = nf[nf[j - 1] - 1]

    vox = np.argwhere(excurset) + 1
    ivox = np.argwhere(np.in1d(vox, lmvox)) + 1
    clmid = nf[ivox - 1]
    uclmid, iclmid, jclmid = np.unique(clmid, return_index=True, return_inverse=True)
    iclmid = iclmid + 1
    jclmid = jclmid + 1
    ucid = np.unique(nf)
    nclus = len(ucid)
    # implementing matlab's histc function ###
    bin_edges = np.r_[-np.Inf, 0.5 * (ucid[:-1] + ucid[1:]), np.Inf]
    ucvol, ucvol_edges = np.histogram(nf, bin_edges)

    if reselspvert is None:
        reselsvox = np.ones(np.shape(vox))
    else:
        reselsvox = reselspvert[vox - 1]

    # calling matlab-python version for scipy's interp1d
    nf1 = interp1(np.append(0, ucid), np.arange(0, nclus + 1), nf, kind="nearest")

    # if k>1, find volume of cluster in added sphere
    if self.k is None or self.k == 1:
        ucrsl = np.bincount(nf1.astype(int), reselsvox.flatten())
    if self.k == 2:
        if l == 1:
            ndf = np.array([self.df]).size
            r = 2 * np.arccos((thresh / slm_t[0, vox - 1]) ** (float(1) / ndf))
        else:
            r = 2 * np.arccos(
                np.sqrt(
                    (thresh - slm_t[1, vox - 1])
                    * (thresh >= slm_t[1, vox - 1])
                    / (slm_t[0, vox - 1] - slm_t[1, vox - 1])
                )
            )
        ucrsl = np.bincount(nf1.astype(int), (r.T * reselsvox.T).flatten())
    if self.k == 3:
        if l == 1:
            ndf = np.array([self.df]).size
            r = 2 * math.pi * (1 - (thresh / slm_t[0, vox - 1]) ** (float(1) / ndf))
        else:
            nt = 20
            theta = (np.arange(1, nt + 1, 1) - 1 / 2) / nt * math.pi / 2
            s = (np.cos(theta) ** 2 * slm_t[1, vox - 1]).T
            if l == 3:
                s = s + ((np.sin(theta) ** 2) * slm_t[2, vox - 1]).T
            r = (
                2
                * math.pi
                * (
                    1
                    - np.sqrt(
                        (thresh - s)
                        * (thresh >= s)
                        / (np.ones((nt, 1)) * slm_t[0, vox - 1].T - s)
                    )
                ).mean(axis=0)
            )
        ucrsl = np.bincount(nf1.astype(int), (r.T * reselsvox.T).flatten())

    # and their ranks (in ascending order)
    iucrls = sorted(range(len(ucrsl[1:])), key=lambda k: ucrsl[1:][k])
    rankrsl = np.zeros((1, nclus))
    rankrsl[0, iucrls] = np.arange(nclus, 0, -1)

    lmid = lmvox[ismember(lmvox, vox)[0]]

    varA = slm_t[0, (lmid - 1)]
    varB = lmid
    varC = rankrsl[0, jclmid - 1]
    varALL = np.concatenate(
        (
            varA.reshape(len(varA), 1),
            varB.reshape(len(varB), 1),
            varC.reshape(len(varC), 1),
        ),
        axis=1,
    )
    lm = np.flipud(
        varALL[
            varALL[:, 0].argsort(),
        ]
    )
    varNEW = np.concatenate(
        (rankrsl.T, ucvol.reshape(len(ucvol), 1), ucrsl.reshape(len(ucrsl), 1)[1:]),
        axis=1,
    )
    cl = varNEW[
        varNEW[:, 0].argsort(),
    ]
    clusid = np.zeros((1, v))
    clusid[0, (vox - 1).T] = interp1(
        np.append(0, ucid), np.append(0, rankrsl), nf, kind="nearest"
    )
    peak = {}
    peak["t"] = lm[:, 0].reshape(len(lm[:, 0]), 1)
    peak["vertid"] = lm[:, 1].reshape(len(lm[:, 1]), 1)
    peak["clusid"] = lm[:, 2].reshape(len(lm[:, 2]), 1)
    clus = {}
    clus["clusid"] = cl[:, 0].reshape(len(cl[:, 0]), 1)
    clus["nverts"] = cl[:, 1].reshape(len(cl[:, 1]), 1)
    clus["resels"] = cl[:, 2].reshape(len(cl[:, 2]), 1)

    return peak, clus, clusid


def compute_resels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SurfStatResels of surface or volume data inside a mask.

    Parameters
    ----------
    slm : a dictionary with keys 'lat' or 'tri' and, optionally, 'resl'.
        slm['lat'] : a 3D numpy array of 1's and 0's.
        slm['tri'] : a 2D numpy array of shape (t, 3).
            Contains triangles of a surface. slm['tri'].max() is equal to the
            number of vertices.
        slm['resl'] : a 2D numpy array of shape (e, k).
            Sum over observations of squares of differences of normalized
            residuals along each edge.
    mask : a 1D numpy array of shape (v), dtype 'bool'.
        v must be equal to int(slm['tri'].max()).
        Contains 1's and 0's (1's are included and 0's are excluded).

    Returns
    -------
    resels : a 2D numpy array of shape (1, (D+1)).
        Array of 0,...,D dimensional resels of the mask, EC of the mask
        if slm['resl'] is not given.
    reselspvert : a 1D numpy array of shape (v).
        Array of D-dimensional resels per mask vertex.
    edg : a 2D numpy array of shape (e, 2).
        Array of edge indices.
    """

    def pacos(x: ArrayLike) -> np.ndarray:
        return np.arccos(np.minimum(np.abs(x), 1) * np.sign(x))

    if self.tri is not None:
        # Get unique edges. Subtract 1 from edges to conform to Python's
        # counting from 0 - RV
        tri = np.sort(self.tri) - 1
        edg = np.unique(
            np.vstack((tri[:, (0, 1)], tri[:, (0, 2)], tri[:, (1, 2)])), axis=0
        )

        # If no mask is provided, create one with all included vertices set to
        # 1. If mask is provided, simply grab the number of vertices from mask.
        if self.mask is None:
            v = np.amax(edg) + 1
            self.mask = np.full(v, False)
            self.mask[edg] = True
        else:
            # if np.ndim(mask) > 1:
            #    mask = np.squeeze(mask)
            #    if mask.shape[0] > 1:
            #        mask = mask.T
            v = self.mask.size

        # Compute the Lipschitz–Killing curvatures (LKC)
        m = np.sum(self.mask)
        if self.resl is not None:
            lkc = np.zeros((3, 3))
        else:
            lkc = np.zeros((1, 3))
        lkc[0, 0] = m

        # LKC of edges
        maskedg = np.all(self.mask[edg], axis=1)
        lkc[0, 1] = np.sum(maskedg)

        if self.resl is not None:
            r1 = np.mean(np.sqrt(self.resl[maskedg, :]), axis=1)
            lkc[1, 1] = np.sum(r1)
        # LKC of triangles
        # Made an adjustment from the MATLAB implementation:
        # The reselspvert computation is included in the if-statement.
        # MATLAB errors when the if statement is false as variable r2 is not
        # defined during the computation of reselspvert. - RV
        masktri = np.all(self.mask[tri], 1)
        lkc[0, 2] = np.sum(masktri)
        if self.resl is not None:
            loc = row_ismember(tri[masktri, :][:, [0, 1]], edg)
            l12 = self.resl[loc, :]
            loc = row_ismember(tri[masktri, :][:, [0, 2]], edg)
            l13 = self.resl[loc, :]
            loc = row_ismember(tri[masktri, :][:, [1, 2]], edg)
            l23 = self.resl[loc, :]
            a = np.fmax(4 * l12 * l13 - (l12 + l13 - l23) ** 2, 0)
            r2 = np.mean(np.sqrt(a), axis=1) / 4
            lkc[1, 2] = (
                np.sum(np.mean(np.sqrt(l12) + np.sqrt(l13) + np.sqrt(l23), axis=1)) / 2
            )
            lkc[2, 2] = np.nansum(r2, axis=0)

            # Compute resels per mask vertex
            reselspvert = np.zeros(v)
            for j in range(0, 3):
                reselspvert = reselspvert + np.bincount(
                    tri[masktri, j], weights=r2, minlength=v
                )
            D = 2
            reselspvert = reselspvert / (D + 1) / np.sqrt(4 * np.log(2)) ** D
        else:
            reselspvert = None

    if self.lat is not None:
        edg = mesh_edges(self)
        # The lattice is filled with 5 alternating tetrahedra per cube
        I, J, K = np.shape(self.lat)
        IJ = I * J
        i, j = np.meshgrid(range(1, I + 1), range(1, J + 1))
        i = np.squeeze(np.reshape(i, (-1, 1)))
        j = np.squeeze(np.reshape(j, (-1, 1)))

        c1 = np.argwhere((((i + j) % 2) == 0) & (i < I) & (j < J))
        c2 = np.argwhere((((i + j) % 2) == 0) & (i > 1) & (j < J))
        c11 = np.argwhere((((i + j) % 2) == 0) & (i == I) & (j < J))
        c21 = np.argwhere((((i + j) % 2) == 0) & (i == I) & (j > 1))
        c12 = np.argwhere((((i + j) % 2) == 0) & (i < I) & (j == J))
        c22 = np.argwhere((((i + j) % 2) == 0) & (i > 1) & (j == J))

        # outcome is 1 lower than MATLAB due to 0-1 counting difference. - RV
        d1 = np.argwhere((((i + j) % 2) == 1) & (i < I) & (j < J)) + IJ
        d2 = np.argwhere((((i + j) % 2) == 1) & (i > 1) & (j < J)) + IJ

        tri1 = cat(
            (
                cat((c1, c1 + 1, c1 + 1 + I), axis=1),
                cat((c1, c1 + I, c1 + 1 + I), axis=1),
                cat((c2 - 1, c2, c2 - 1 + I), axis=1),
                cat((c2, c2 - 1 + I, c2 + I), axis=1),
            ),
            axis=0,
        )
        tri2 = cat(
            (
                cat((c1, c1 + 1, c1 + 1 + IJ), axis=1),
                cat((c1, c1 + IJ, c1 + 1 + IJ), axis=1),
                cat((c1, c1 + I, c1 + I + IJ), axis=1),
                cat((c1, c1 + IJ, c1 + I + IJ), axis=1),
                cat((c1, c1 + 1 + I, c1 + 1 + IJ), axis=1),
                cat((c1, c1 + 1 + I, c1 + I + IJ), axis=1),
                cat((c1, c1 + 1 + IJ, c1 + I + IJ), axis=1),
                cat((c1 + 1 + I, c1 + 1 + IJ, c1 + I + IJ), axis=1),
                cat((c2 - 1, c2, c2 - 1 + IJ), axis=1),
                cat((c2, c2 - 1 + IJ, c2 + IJ), axis=1),
                cat((c2 - 1, c2 - 1 + I, c2 - 1 + IJ), axis=1),
                cat((c2 - 1 + I, c2 - 1 + IJ, c2 - 1 + I + IJ), axis=1),
                cat((c2, c2 - 1 + I, c2 + I + IJ), axis=1),
                cat((c2, c2 - 1 + IJ, c2 + I + IJ), axis=1),
                cat((c2, c2 - 1 + I, c2 - 1 + IJ), axis=1),
                cat((c2 - 1 + I, c2 - 1 + IJ, c2 + I + IJ), axis=1),
                cat((c11, c11 + I, c11 + I + IJ), axis=1),
                cat((c11, c11 + IJ, c11 + I + IJ), axis=1),
                cat((c21 - I, c21, c21 - I + IJ), axis=1),
                cat((c21, c21 - I + IJ, c21 + IJ), axis=1),
                cat((c12, c12 + 1, c12 + 1 + IJ), axis=1),
                cat((c12, c12 + IJ, c12 + 1 + IJ), axis=1),
                cat((c22 - 1, c22, c22 - 1 + IJ), axis=1),
                cat((c22, c22 - 1 + IJ, c22 + IJ), axis=1),
            ),
            axis=0,
        )
        tri3 = cat(
            (
                cat((d1, d1 + 1, d1 + 1 + I), axis=1),
                cat((d1, d1 + I, d1 + 1 + I), axis=1),
                cat((d2 - 1, d2, d2 - 1 + I), axis=1),
                cat((d2, d2 - 1 + I, d2 + I), axis=1),
            ),
            axis=0,
        )
        tet1 = cat(
            (
                cat((c1, c1 + 1, c1 + 1 + I, c1 + 1 + IJ), axis=1),
                cat((c1, c1 + I, c1 + 1 + I, c1 + I + IJ), axis=1),
                cat((c1, c1 + 1 + I, c1 + 1 + IJ, c1 + I + IJ), axis=1),
                cat((c1, c1 + IJ, c1 + 1 + IJ, c1 + I + IJ), axis=1),
                cat((c1 + 1 + I, c1 + 1 + IJ, c1 + I + IJ, c1 + 1 + I + IJ), axis=1),
                cat((c2 - 1, c2, c2 - 1 + I, c2 - 1 + IJ), axis=1),
                cat((c2, c2 - 1 + I, c2 + I, c2 + I + IJ), axis=1),
                cat((c2, c2 - 1 + I, c2 - 1 + IJ, c2 + I + IJ), axis=1),
                cat((c2, c2 - 1 + IJ, c2 + IJ, c2 + I + IJ), axis=1),
                cat((c2 - 1 + I, c2 - 1 + IJ, c2 - 1 + I + IJ, c2 + I + IJ), axis=1),
            ),
            axis=0,
        )

        v = np.int(np.round(np.sum(self.lat)))
        if self.mask is None:
            self.mask = np.ones(v, dtype=bool)

        reselspvert = np.zeros(v)
        vs = np.cumsum(np.squeeze(np.sum(np.sum(self.lat, axis=0), axis=0)))
        vs = cat((np.zeros(1), vs, np.expand_dims(vs[K - 1], axis=0)), axis=0)
        vs = vs.astype(int)
        es = 0
        lat = np.zeros((I, J, 2))
        lat[:, :, 0] = self.lat[:, :, 0]
        lkc = np.zeros((4, 4))
        for k in range(0, K):
            f = (k + 1) % 2
            if k < (K - 1):
                lat[:, :, f] = self.lat[:, :, k + 1]
            else:
                lat[:, :, f] = np.zeros((I, J))
            vid = (np.cumsum(lat.flatten("F")) * np.reshape(lat.T, -1)).astype(int)
            if f:
                edg1 = (
                    edg[
                        np.logical_and(
                            edg[:, 0] > (vs[k] - 1), edg[:, 0] <= (vs[k + 1] - 1)
                        ),
                        :,
                    ]
                    - vs[k]
                )
                edg2 = (
                    edg[
                        np.logical_and(
                            edg[:, 0] > (vs[k] - 1), edg[:, 1] <= (vs[k + 2] - 1)
                        ),
                        :,
                    ]
                    - vs[k]
                )
                # Added a -1 - RV
                tri = (
                    cat(
                        (
                            vid[
                                tri1[
                                    np.all(
                                        np.reshape(lat.flatten("F")[tri1], tri1.shape),
                                        1,
                                    ),
                                    :,
                                ]
                            ],
                            vid[
                                tri2[
                                    np.all(
                                        np.reshape(lat.flatten("F")[tri2], tri2.shape),
                                        1,
                                    ),
                                    :,
                                ]
                            ],
                        ),
                        axis=0,
                    )
                    - 1
                )
                mask1 = self.mask[np.arange(vs[k], vs[k + 2])]
            else:
                edg1 = cat(
                    (
                        edg[
                            np.logical_and(
                                edg[:, 0] > (vs[k] - 1), edg[:, 1] <= (vs[k + 1] - 1)
                            ),
                            :,
                        ]
                        - vs[k]
                        + vs[k + 2]
                        - vs[k + 1],
                        cat(
                            (
                                np.expand_dims(
                                    edg[
                                        np.logical_and(
                                            edg[:, 0] <= (vs[k + 1] - 1),
                                            edg[:, 1] > (vs[k + 1] - 1),
                                        ),
                                        1,
                                    ]
                                    - vs[k + 1],
                                    axis=1,
                                ),
                                np.expand_dims(
                                    edg[
                                        np.logical_and(
                                            edg[:, 0] <= (vs[k + 1] - 1),
                                            edg[:, 1] > (vs[k + 1] - 1),
                                        ),
                                        0,
                                    ]
                                    - vs[k]
                                    + vs[k + 2]
                                    - vs[k + 1],
                                    axis=1,
                                ),
                            ),
                            axis=1,
                        ),
                    ),
                    axis=0,
                )
                edg2 = cat(
                    (
                        edg1,
                        edg[
                            np.logical_and(
                                edg[:, 0] > (vs[k + 1] - 1),
                                edg[:, 1] <= (vs[k + 2] - 1),
                            ),
                            :,
                        ]
                        - vs[k + 1],
                    ),
                    axis=0,
                )
                # Added a -1 - RV
                tri = (
                    cat(
                        (
                            vid[tri3[np.all(lat.flatten("F")[tri3], axis=1), :]],
                            vid[tri2[np.all(lat.flatten("F")[tri2], axis=1), :]],
                        ),
                        axis=0,
                    )
                    - 1
                )
                mask1 = cat(
                    (
                        self.mask[np.arange(vs[k + 1], vs[k + 2])],
                        self.mask[np.arange(vs[k], vs[k + 1])],
                    )
                )
            # Added a -1 -RV
            tet = vid[tet1[np.all(lat.flatten("F")[tet1], axis=1), :]] - 1
            m1 = np.max(edg2[:, 0])
            ue = edg2[:, 0] + m1 * (edg2[:, 1] - 1)
            e = edg2.shape[0]
            ae = np.arange(0, e)
            if e < 2 ** 31:
                sparsedg = csr_matrix(
                    (ae, (ue, np.zeros(ue.shape, dtype=int))), dtype=np.int
                )
                sparsedg.eliminate_zeros()
            ##
            lkc1 = np.zeros((4, 4))
            lkc1[0, 0] = np.sum(self.mask[np.arange(vs[k], vs[k + 1])])

            # LKC of edges
            maskedg = np.all(mask1[edg1], axis=1)

            lkc1[0, 1] = np.sum(maskedg)
            if self.resl is not None:
                r1 = np.mean(np.sqrt(self.resl[np.argwhere(maskedg) + es, :]), axis=1)
                lkc1[1, 1] = np.sum(r1)

            # LKC of triangles
            masktri = np.all(mask1[tri], axis=1).flatten()
            lkc1[0, 2] = np.sum(masktri)
            if self.resl is not None:
                if all(masktri == False):
                    # Set these variables to empty arrays to match the MATLAB
                    # implementation.
                    lkc1[1, 2] = 0
                    lkc1[2, 2] = 0
                else:
                    if e < 2 ** 31:
                        l12 = self.resl[
                            sparsedg[
                                tri[masktri, 0] + m1 * (tri[masktri, 1] - 1), 0
                            ].toarray()
                            + es,
                            :,
                        ]
                        l13 = self.resl[
                            sparsedg[
                                tri[masktri, 0] + m1 * (tri[masktri, 2] - 1), 0
                            ].toarray()
                            + es,
                            :,
                        ]
                        l23 = self.resl[
                            sparsedg[
                                tri[masktri, 1] + m1 * (tri[masktri, 2] - 1), 0
                            ].toarray()
                            + es,
                            :,
                        ]
                    else:
                        l12 = self.resl[
                            interp1(
                                ue,
                                ae,
                                tri[masktri, 0] + m1 * (tri[masktri, 1] - 1),
                                kind="nearest",
                            )
                            + es,
                            :,
                        ]
                        l13 = self.resl[
                            interp1(
                                ue,
                                ae,
                                tri[masktri, 0] + m1 * (tri[masktri, 2] - 1),
                                kind="nearest",
                            )
                            + es,
                            :,
                        ]
                        l23 = self.resl[
                            interp1(
                                ue,
                                ae,
                                tri[masktri, 1] + m1 * (tri[masktri, 2] - 1),
                                kind="nearest",
                            )
                            + es,
                            :,
                        ]
                    a = np.fmax(4 * l12 * l13 - (l12 + l13 - l23) ** 2, 0)
                    r2 = np.mean(np.sqrt(a), axis=1) / 4
                    lkc1[1, 2] = (
                        np.sum(
                            np.mean(np.sqrt(l12) + np.sqrt(l13) + np.sqrt(l23), axis=1)
                        )
                        / 2
                    )
                    lkc1[2, 2] = np.sum(r2)

                # The following if-statement has nargout >=2 in MATLAB,
                # but there's no Python equivalent so ignore that. - RV
                if K == 1:
                    for j in range(0, 3):
                        if f:
                            v1 = tri[masktri, j] + vs[k]
                        else:
                            v1 = tri[masktri, j] + vs[k + 1]
                            v1 = v1 - int(vs > vs[k + 2]) * (vs[k + 2] - vs[k])
                        reselspvert += np.bincount(v1, r2, v)

            # LKC of tetrahedra
            masktet = np.all(mask1[tet], axis=1).flatten()
            lkc1[0, 3] = np.sum(masktet)
            if self.resl is not None and k < (K - 1):
                if e < 2 ** 31:
                    l12 = self.resl[
                        (
                            sparsedg[
                                tet[masktet, 0] + m1 * (tet[masktet, 1] - 1), 0
                            ].toarray()
                            + es
                        ).tolist(),
                        :,
                    ]
                    l13 = self.resl[
                        (
                            sparsedg[
                                tet[masktet, 0] + m1 * (tet[masktet, 2] - 1), 0
                            ].toarray()
                            + es
                        ).tolist(),
                        :,
                    ]
                    l23 = self.resl[
                        (
                            sparsedg[
                                tet[masktet, 1] + m1 * (tet[masktet, 2] - 1), 0
                            ].toarray()
                            + es
                        ).tolist(),
                        :,
                    ]
                    l14 = self.resl[
                        (
                            sparsedg[
                                tet[masktet, 0] + m1 * (tet[masktet, 3] - 1), 0
                            ].toarray()
                            + es
                        ).tolist(),
                        :,
                    ]
                    l24 = self.resl[
                        (
                            sparsedg[
                                tet[masktet, 1] + m1 * (tet[masktet, 3] - 1), 0
                            ].toarray()
                            + es
                        ).tolist(),
                        :,
                    ]
                    l34 = self.resl[
                        (
                            sparsedg[
                                tet[masktet, 2] + m1 * (tet[masktet, 3] - 1), 0
                            ].toarray()
                            + es
                        ).tolist(),
                        :,
                    ]
                else:
                    l12 = self.resl[
                        interp1(
                            ue,
                            ae,
                            tet[masktet, 0] + m1 * (tet[masktet, 1] - 1),
                            kind="nearest",
                        )
                        + es,
                        :,
                    ]
                    l13 = self.resl[
                        interp1(
                            ue,
                            ae,
                            tet[masktet, 0] + m1 * (tet[masktet, 2] - 1),
                            kind="nearest",
                        )
                        + es,
                        :,
                    ]
                    l23 = self.resl[
                        interp1(
                            ue,
                            ae,
                            tet[masktet, 1] + m1 * (tet[masktet, 2] - 1),
                            kind="nearest",
                        )
                        + es,
                        :,
                    ]
                    l14 = self.resl[
                        interp1(
                            ue,
                            ae,
                            tet[masktet, 0] + m1 * (tet[masktet, 3] - 1),
                            kind="nearest",
                        )
                        + es,
                        :,
                    ]
                    l24 = self.resl[
                        interp1(
                            ue,
                            ae,
                            tet[masktet, 1] + m1 * (tet[masktet, 3] - 1),
                            kind="nearest",
                        )
                        + es,
                        :,
                    ]
                    l34 = self.resl[
                        interp1(
                            ue,
                            ae,
                            tet[masktet, 2] + m1 * (tet[masktet, 3] - 1),
                            kind="nearest",
                        )
                        + es,
                        :,
                    ]
                a4 = np.fmax(4 * l12 * l13 - (l12 + l13 - l23) ** 2, 0)
                a3 = np.fmax(4 * l12 * l14 - (l12 + l14 - l24) ** 2, 0)
                a2 = np.fmax(4 * l13 * l14 - (l13 + l14 - l34) ** 2, 0)
                a1 = np.fmax(4 * l23 * l24 - (l23 + l24 - l34) ** 2, 0)

                d12 = 4 * l12 * l34 - (l13 + l24 - l23 - l14) ** 2
                d13 = 4 * l13 * l24 - (l12 + l34 - l23 - l14) ** 2
                d14 = 4 * l14 * l23 - (l12 + l34 - l24 - l13) ** 2

                h = np.logical_or(a1 <= 0, a2 <= 0)
                delta12 = np.sum(
                    np.mean(
                        np.sqrt(l34)
                        * pacos(
                            (d12 - a1 - a2) / np.sqrt(a1 * a2 + h) / 2 * (1 - h) + h
                        ),
                        axis=1,
                    )
                )
                h = np.logical_or(a1 <= 0, a3 <= 0)
                delta13 = np.sum(
                    np.mean(
                        np.sqrt(l24)
                        * pacos(
                            (d13 - a1 - a3) / np.sqrt(a1 * a3 + h) / 2 * (1 - h) + h
                        ),
                        axis=1,
                    )
                )
                h = np.logical_or(a1 <= 0, a4 <= 0)
                delta14 = np.sum(
                    np.mean(
                        np.sqrt(l23)
                        * pacos(
                            (d14 - a1 - a4) / np.sqrt(a1 * a4 + h) / 2 * (1 - h) + h
                        ),
                        axis=1,
                    )
                )
                h = np.logical_or(a2 <= 0, a3 <= 0)
                delta23 = np.sum(
                    np.mean(
                        np.sqrt(l14)
                        * pacos(
                            (d14 - a2 - a3) / np.sqrt(a2 * a3 + h) / 2 * (1 - h) + h
                        ),
                        axis=1,
                    )
                )
                h = np.logical_or(a2 <= 0, a4 <= 0)
                delta24 = np.sum(
                    np.mean(
                        np.sqrt(l13)
                        * pacos(
                            (d13 - a2 - a4) / np.sqrt(a2 * a4 + h) / 2 * (1 - h) + h
                        ),
                        axis=1,
                    )
                )
                h = np.logical_or(a3 <= 0, a4 <= 0)
                delta34 = np.sum(
                    np.mean(
                        np.sqrt(l12)
                        * pacos(
                            (d12 - a3 - a4) / np.sqrt(a3 * a4 + h) / 2 * (1 - h) + h
                        ),
                        axis=1,
                    )
                )

                r3 = np.squeeze(
                    np.mean(
                        np.sqrt(
                            np.fmax(
                                (4 * a1 * a2 - (a1 + a2 - d12) ** 2)
                                / (l34 + (l34 <= 0))
                                * (l34 > 0),
                                0,
                            )
                        ),
                        axis=1,
                    )
                    / 48
                )

                lkc1[1, 3] = (
                    delta12 + delta13 + delta14 + delta23 + delta24 + delta34
                ) / (2 * np.pi)
                lkc1[2, 3] = (
                    np.sum(
                        np.mean(
                            np.sqrt(a1) + np.sqrt(a2) + np.sqrt(a3) + np.sqrt(a4),
                            axis=1,
                        )
                    )
                    / 8
                )
                lkc1[3, 3] = np.sum(r3)

                # Original MATLAB code has a if nargout>=2 here, ignore it
                # as no equivalent exists in Python - RV.
                for j in range(0, 4):
                    if f:
                        v1 = tet[masktet, j] + vs[k]
                    else:
                        v1 = tet[masktet, j] + vs[k + 1]
                        v1 = v1 - (v1 > (vs[k + 2] - 1)) * (vs[k + 2] - vs[k])
                    if np.ndim(r3) == 0:
                        r3 = r3.tolist()
                        r3 = [r3]
                    reselspvert += np.bincount(v1, r3, v)
            lkc = lkc + lkc1
            es = es + edg1.shape[0]

        # Original MATLAB code has a if nargout>=2 here,
        # ignore it as no equivalent exists in Python - RV.
        D = 2 + (K > 1)
        reselspvert = reselspvert / (D + 1) / np.sqrt(4 * np.log(2)) ** D

    # Compute resels - RV
    D1 = lkc.shape[0] - 1
    D2 = lkc.shape[1] - 1
    tpltz = toeplitz((-1) ** (np.arange(0, D1 + 1)), (-1) ** (np.arange(0, D2 + 1)))
    lkcs = np.sum(tpltz * lkc, axis=1).T
    lkcs = np.trim_zeros(lkcs, trim="b")
    lkcs = np.atleast_2d(lkcs)
    D = lkcs.shape[1] - 1
    resels = lkcs / np.sqrt(4 * np.log(2)) ** np.arange(0, D + 1)

    return resels, reselspvert, edg
