from sklearn.base import BaseEstimator
import warnings
import numpy as np
from cmath import sqrt
from .terms import Term


class SLM(BaseEstimator):
    # Import class methods
    from ._models import linear_model, t_test
    from ._multiple_comparisons import fdr, random_field_theory

    def __init__(
        self,
        model,
        contrast,
        surf=None,
        *,
        correction=None,
        niter=1,
        thetalim=0.01,
        drlim=0.1,
        one_tailed=False,
        cluster_threshold=0.001,
        mask=None,
    ):
        # Input arguments.
        self.model = model
        self.contrast = contrast
        self.surf = surf
        self.correction = correction
        self.niter = niter
        self.thetalim = thetalim
        self.drlim = drlim
        self.one_tailed = one_tailed
        self.cluster_threshold = cluster_threshold
        self.mask = mask

        # Parameters created by functions.
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

    def fit(self, Y, mask=None):
        self.linear_model(Y)
        self.t_test()
        if self._correction is "rft":
            self.P = self.random_field_theory(mask, self.cluster_threshold)
        elif self._correction is "fdr":
            self.Q = self.fdr(mask)

    # TODO: Add onetailed/twotailed


def f_test(slm1, slm2):
    """F-statistics for comparing two uni- or multi-variate fixed effects models.

    Parameters
    ----------
    slm1 : dict
        Standard linear model returned by the t_test function; see Notes for
        details.
    slm2 : dict
        Standard linear model returned by the t_test function; see Notes for
        details.

    Returns
    -------
    slm : dict
        Standard linear model; see Notes for details.

    See Also
    --------
    brainstat.stats.models.t_test : Computes t-values for a linear model.

    Notes
    ------
    The slm1 and slm2 dictionaries must contain the following fields:

    - slm1['X'] : (numpy.array) the design matrix.
    - slm1['df'] (numpy.array, int) the degrees of freedom.
    - slm1['SSE'] : (numpy.array) sum of squares of errors.
    - slm1['coef'] : (numpy.array) coefficients of the linear model.

    Fields of the bigger model are copied to the ouput slm, and the following
    fields are added/altered:

    - slm['k'] : (numpy.array) Number of variates.
    - slm['df'] : (numpy.array) two-element vector containing [df1-df2, df2] where df1 and df2 are the min/max of the input dfs.
    - slm['t'] : Matrix of non-zero eigenvalues, in descending order, derived using Roy's maximum root.

    """

    if slm1.r is not None or slm2.r is not None:
        warnings.warn("Mixed effects models not programmed yet.")

    slm = SLM(Term(1), Term(1))
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
            setattr(slm, key, getattr(slm2, key))

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
            print("Roy" "s max root for k>3 not programmed yet.")
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
