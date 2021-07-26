""" Standard Linear regression models. """
import warnings
from cmath import sqrt

import numpy as np
from brainspace.mesh.mesh_elements import get_cells, get_points
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat.mesh.utils import _mask_edges, mesh_edges

from .terms import FixedEffect
from .utils import apply_mask, undo_mask


class SLM:
    """Core Class for running BrainStat linear models"""

    # Import class methods
    from ._linear_model import linear_model
    from ._multiple_comparisons import fdr, random_field_theory
    from ._t_test import t_test

    def __init__(
        self,
        model,
        contrast,
        surf=None,
        mask=None,
        *,
        correction=None,
        niter=1,
        thetalim=0.01,
        drlim=0.1,
        two_tailed=True,
        cluster_threshold=0.001,
    ):
        """Constructor for the SLM class.

        Parameters
        ----------
        model : brainstat.stats.terms.FixedEffect, brainstat.stats.terms.MixedEffect
            The linear model to be fitted of dimensions (observations, predictors).
        contrast : array-like, brainstat.stats.terms.FixedEffect
            Vector of contrasts in the observations.
        surf : dict, BSPolyData, optional
            A surface provided as either a dictionary with keys 'tri' for its
            faces (n-by-3 array) and 'coord' for its coordinates (3-by-n array),
            or as a BrainSpace BSPolyData object by default None.
        mask : array-like, optional
            A mask containing True for vertices to include in the analysis, by
            default None.
        correction : str, list, optional
            String or list of strings. If it contains "rft" a random field
            theory multiple comparisons correction will be run. If it contains
            "fdr" a false discovery rate multiple comparisons correction will be
            run. Both may be provided. By default None.
        niter : int, optional
            Number of iterations of the Fisher scoring algorithm for fitting
            mixed effects models, by default 1.
        thetalim : float, optional
            Lower limit on variance coefficients in standard deviations, by
            default 0.01.
        drlim : float, optional
            Step of ratio of variance coefficients in standard deviations, by
            default 0.1.
        two_tailed : bool, optional
            Determines whether to return two-tailed or one-tailed p-values. Note
            that multivariate analyses can only be two-tailed, by default True.
        cluster_threshold : float, optional
            P-value threshold or statistic threshold for defining clusters in
            random field theory, by default 0.001.
        """
        # Input arguments.
        self.model = model
        self.contrast = contrast
        self.surf = surf
        self.mask = mask
        self.correction = correction
        if isinstance(self.correction, str):
            self.correction = [self.correction]
        self.niter = niter
        self.thetalim = thetalim
        self.drlim = drlim
        self.two_tailed = two_tailed
        self.cluster_threshold = cluster_threshold

        # Error check
        if self.surf is None:
            if self.correction is not None and "rft" in self.correction:
                raise ValueError("Random Field Theory corrections require a surface.")

        # We have to initialize fit parameters for our unit tests here.
        # TODO: remove this requirement.
        self._reset_fit_parameters()

    def fit(self, Y):
        """Fits the SLM model

        Parameters
        ----------
        Y : numpy.array
            Input data (observation, vertex, variate)

        Raises
        ------
        ValueError
            An error will be thrown when multivariate data is provided and a
            one-tailed test is requested.
        """
        if Y.ndim > 2:
            if (not self.two_tailed) and Y.shape[2] > 1:
                raise ValueError(
                    "One-tailed tests are not implemented for multivariate data."
                )
            student_t_test = Y.shape[2] == 1
        else:
            student_t_test = True

        self._reset_fit_parameters()
        if self.mask is not None:
            Y_masked = apply_mask(Y, self.mask, axis=1)
        else:
            Y_masked = Y
        self.linear_model(Y_masked)
        self.t_test()
        if self.mask is not None:
            self._unmask()
        if self.correction is not None:
            self.multiple_comparison_corrections(student_t_test)

    def multiple_comparison_corrections(self, student_t_test):
        """Performs multiple comparisons corrections. If a (one-sided) student-t
        test was run, then make it two-tailed if requested."""
        P1, Q1 = self._run_multiple_comparisons()

        if self.two_tailed and student_t_test:
            self.t = -self.t
            P2, Q2 = self._run_multiple_comparisons()
            self.t = -self.t
            self.P = _merge_rft(P1, P2)
            self.Q = _merge_fdr(Q1, Q2)
        else:
            self.P = P1
            self.Q = Q1

    def _run_multiple_comparisons(self):
        """Runs the multiple comparisons tests and returns their outputs.

        Returns
        -------
        dict, None
            Results of random_field_theory. None if not requested.
        np.array, None
            Results of fdr. None if not requested.
        """
        P = None
        Q = None
        if "rft" in self.correction:
            P = {}
            P["pval"], P["peak"], P["clus"], P["clusid"] = self.random_field_theory()
        if "fdr" in self.correction:
            Q = self.fdr()
        return P, Q

    def _reset_fit_parameters(self):
        """Sets empty parameters before fitting. Prevents issues arising from
        using the same object to fit twice.
        """
        self.X = None
        self.t = None
        self.df = None
        self.SSE = None
        self.coef = None
        self.V = None
        self.k = None
        self.r = None
        self.dr = None
        self.resl = None
        self.tri = None
        self.lat = None
        self.c = None
        self.ef = None
        self.sd = None
        self.dfs = None
        self.P = None
        self.Q = None
        self.du = None

    def _unmask(self):
        """Changes all masked parameters to their input dimensions."""
        simple_unmask_parameters = ["t", "coef", "SSE", "r", "ef", "sd", "dfs"]
        for key in simple_unmask_parameters:
            attr = getattr(self, key)
            if attr is not None:
                setattr(self, key, undo_mask(attr, self.mask, axis=1))

        # slm.resl unmask
        if self.resl is not None:
            edges = mesh_edges(self.surf)
            _, idx = _mask_edges(edges, self.mask)
            self.resl = undo_mask(self.resl, idx, axis=0)

    """ Property specifications. """

    @property
    def surf(self):
        return self._surf

    @surf.setter
    def surf(self, value):
        self._surf = value
        if self.surf is not None:
            if isinstance(self.surf, BSPolyData):
                self.tri = np.array(get_cells(self.surf)) + 1
                self.coord = np.array(get_points(self.surf)).T
            else:
                if "tri" in value:
                    self.tri = value["tri"]
                    self.coord = value["coord"]
                elif "lat" in value:
                    self.lat = value["lat"]
                    self.coord = value["coord"]

    @surf.deleter
    def surf(self):
        del self._surf

    @property
    def tri(self):
        return self._tri

    @tri.setter
    def tri(self, value):
        if value is not None and np.any(value < 0):
            raise ValueError("Triangle indices must be non-negative.")
        self._tri = value

    @tri.deleter
    def tri(self):
        del self._tri

    @property
    def lat(self):
        return self._lat

    @lat.setter
    def lat(self, value):
        self._lat = value

    @lat.deleter
    def lat(self):
        del self._lat


def _merge_rft(P1, P2):
    """Merge two one-tailed outputs of the random_field_theory function.

    Parameters
    ----------
    P1 : dict
        Output dict of random_field_theory
    P2 : dict
        Output dict of random_field_theory

    Returns
    -------
    dict
        Two-tailed version of the inputs.
    """
    if P1 is None and P2 is None:
        return None

    P = {}
    for key1 in P1:
        P[key1] = {}
        if key1 == "clusid":
            P[key1] = [P1[key1], P2[key1]]
            continue
        for key2 in P1[key1]:
            if key2 == "P" and key1 == "pval":
                P[key1][key2] = _onetailed_to_twotailed(P1[key1][key2], P2[key1][key2])
            else:
                P[key1][key2] = [P1[key1][key2], P2[key1][key2]]
    return P


def _merge_fdr(Q1, Q2):
    """Merge two one-tailed outputs of the fdr function.

    Parameters
    ----------
    Q1 : array-like, None
        Q-values
    Q2 : array-like, None
        Q-values

    Returns
    -------
    array-like
        Two-tailed FDR p-values
    """
    if Q1 is None and Q2 is None:
        return None
    return _onetailed_to_twotailed(Q1, Q2)


def _onetailed_to_twotailed(p1, p2):
    """Converts two one-tailed tests to a two-tailed test"""
    return np.minimum(np.minimum(p1, p2) * 2, 1)


def f_test(slm1, slm2):
    """F-statistics for comparing two uni- or multi-variate fixed effects models.

    Parameters
    ----------
    slm1 : brainstat.stats.SLM.SLM
        Standard linear model returned by the t_test function; see Notes for
        details.
    slm2 : brainstat.stats.SLM.SLM
        Standard linear model returned by the t_test function; see Notes for
        details.

    Returns
    -------
    brainstat.stats.SLM.SLM
        Standard linear model with f-test results included.

    """

    if slm1.r is not None or slm2.r is not None:
        warnings.warn("Mixed effects models not programmed yet.")

    slm = SLM(FixedEffect(1), FixedEffect(1))
    if slm1.df > slm2.df:
        X1 = slm1.X
        X2 = slm2.X
        df1 = slm1.df
        df2 = slm2.df
        SSE1 = slm1.SSE
        SSE2 = slm2.SSE
        for key in slm2.__dict__:
            setattr(slm, key, getattr(slm2, key))
    else:
        X1 = slm2.X
        X2 = slm1.X
        df1 = slm2.df
        df2 = slm1.df
        SSE1 = slm2.SSE
        SSE2 = slm1.SSE
        for key in slm1.__dict__:
            setattr(slm, key, getattr(slm1, key))

    r = X1 - np.dot(np.dot(X2, np.linalg.pinv(X2)), X1)
    d = np.sum(r.flatten() ** 2) / np.sum(X1.flatten() ** 2)

    if d > np.spacing(1):
        print("Models are not nested.")
        return

    slm.df = np.array([[df1 - df2, df2]])
    h = SSE1 - SSE2

    # if slm['coef'] is 3D and third dimension is 1, then squeeze it to 2D
    if np.ndim(slm.coef) == 3 and np.shape(slm.coef)[2] == 1:
        x1, x2, x3 = np.shape(slm.coef)
        slm.coef = slm.coef.reshape(x1, x2)

    if np.ndim(slm.coef) == 2:
        slm.k = np.array(1)
        slm.t = np.dot(h / (SSE2 + (SSE2 <= 0)) * (SSE2 > 0), df2 / (df1 - df2))
    elif np.ndim(slm.coef) > 2:
        k2, v = np.shape(SSE2)
        k = np.around((np.sqrt(1 + 8 * k2) - 1) / 2)
        slm.k = np.array(k)
        if k > 3:
            print("Roy's max root for k>3 not programmed yet.")
            return

        l = min(k, df1 - df2)
        slm.t = np.zeros((int(l), int(v)))

        if k == 2:
            det = SSE2[0, :] * SSE2[2, :] - SSE2[1, :] ** 2
            a11 = SSE2[2, :] * h[0, :] - SSE2[1, :] * h[1, :]
            a21 = SSE2[0, :] * h[1, :] - SSE2[1, :] * h[0, :]
            a12 = SSE2[2, :] * h[1, :] - SSE2[1, :] * h[2, :]
            a22 = SSE2[0, :] * h[2, :] - SSE2[1, :] * h[1, :]
            a0 = a11 * a22 - a12 * a21
            a1 = (a11 + a22) / 2
            s1 = np.array([sqrt(x) for x in (a1 ** 2 - a0)]).real
            d = (df2 / (df1 - df2)) / (det + (det <= 0)) * (det > 0)
            slm.t[0, :] = (a1 + s1) * d
            if l == 2:
                slm.t[1, :] = (a1 - s1) * d
        if k == 3:
            det = (
                SSE2[0, :] * (SSE2[2, :] * SSE2[5, :] - SSE2[4, :] ** 2)
                - SSE2[5, :] * SSE2[1, :] ** 2
                + SSE2[3, :] * (SSE2[1, :] * SSE2[4, :] * 2 - SSE2[2, :] * SSE2[3, :])
            )
            m1 = SSE2[2, :] * SSE2[5, :] - SSE2[4, :] ** 2
            m3 = SSE2[0, :] * SSE2[5, :] - SSE2[3, :] ** 2
            m6 = SSE2[0, :] * SSE2[2, :] - SSE2[1, :] ** 2
            m2 = SSE2[3, :] * SSE2[4, :] - SSE2[1, :] * SSE2[5, :]
            m4 = SSE2[1, :] * SSE2[4, :] - SSE2[2, :] * SSE2[3, :]
            m5 = SSE2[1, :] * SSE2[3, :] - SSE2[0, :] * SSE2[4, :]
            a11 = m1 * h[0, :] + m2 * h[1, :] + m4 * h[3, :]
            a12 = m1 * h[1, :] + m2 * h[2, :] + m4 * h[4, :]
            a13 = m1 * h[3, :] + m2 * h[4, :] + m4 * h[5, :]
            a21 = m2 * h[0, :] + m3 * h[1, :] + m5 * h[3, :]
            a22 = m2 * h[1, :] + m3 * h[2, :] + m5 * h[4, :]
            a23 = m2 * h[3, :] + m3 * h[4, :] + m5 * h[5, :]
            a31 = m4 * h[0, :] + m5 * h[1, :] + m6 * h[3, :]
            a32 = m4 * h[1, :] + m5 * h[2, :] + m6 * h[4, :]
            a33 = m4 * h[3, :] + m5 * h[4, :] + m6 * h[5, :]
            a0 = (
                -a11 * (a22 * a33 - a23 * a32)
                + a12 * (a21 * a33 - a23 * a31)
                - a13 * (a21 * a32 - a22 * a31)
            )
            a1 = a22 * a33 - a23 * a32 + a11 * a33 - a13 * a31 + a11 * a22 - a12 * a21
            a2 = -(a11 + a22 + a33)
            q = a1 / 3 - a2 ** 2 / 9
            r = (a1 * a2 - 3 * a0) / 6 - a2 ** 3 / 27
            s1 = (r + [sqrt(x) for x in (q ** 3 + r ** 2)]) ** (1 / 3)
            z = np.zeros((3, v))
            z[0, :] = 2 * s1.real - a2 / 3
            z[1, :] = -s1.real - a2 / 3 + np.sqrt(3) * s1.imag
            z[2, :] = -s1.real - a2 / 3 - np.sqrt(3) * s1.imag

            if not np.count_nonzero(z) == 0:
                z.sort(axis=0)
                z = z[::-1]
            d = df2 / (df1 - df2) / (det + (det <= 0)) * (det > 0)

            for j in range(0, l):
                slm.t[j, :] = z[j, :] * d
    return slm
