import warnings
import math
import numpy as np
import numpy.linalg as la
import scipy
import sys
from scipy.linalg import null_space
from scipy.linalg import cholesky
from ..mesh.utils import mesh_edges
from brainstat.stats.terms import Term, Random
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainspace.mesh.mesh_elements import get_cells


class SLM(object):
    # Import class methods
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
        one_tailed=True,
        cluster_threshold=0.001,
    ):
        # Input arguments.
        self._model = model
        self._contrast = contrast
        self._surf = surf
        self._correction = correction
        self._niter = niter
        self._thetalim = thetalim
        self._drlim = drlim
        self._one_tailed = one_tailed
        self._cluster_threshold=cluster_threshold

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
        if self._correction is 'rft':
            self.P = self.random_field_theory(mask, self.cluster_threshold)
        elif self._correction is 'fdr':
            self.Q = self.fdr(mask)

    # Setters/Getters for input arguments where we want validation.
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, x):
        if isinstance(x, Term) or isinstance(x, Random):
            self._model = x
        else:
            ValueError("Input model must be a Term or Random class.")
    
    @property
    def correction(self):
        return self._correction
    @correction.setter
    def corection(self, x):
        if x in ['fdr', 'rft'] or x is None:
            self._correction = x
        else:
            ValueError('Unknown multiple comparisons correction method.')
    

    def linear_model(self, Y):
        """Fits linear mixed effects models to surface data and estimates resels.

        Parameters
        ----------
        Y : numpy array
            Input data of shape (samples, vertices, features).
        M : Term, Random
            Design matrix.
        surf : dict, BSPolyData, optional
            Surface triangles (surf['tri']) or volumetric data (surf['lat']).
            If 'tri', shape = (n_edges, 2). If 'lat', then it is a boolean 3D
            array. Alternatively a BSPolyData object can be provided. Default is None.
        niter : int, optional
            Number of extra iterations of the Fisher scoring algorithm for fitting
            mixed effects models. Default is 1.
        thetalim : float, optional
            Lower limit on variance coefficients, in sd's. Default is 0.01.
        drlim : float, optional
            Step of ratio of variance coefficients, in sd's. Default 0.1.

        Returns
        -------
        slm : dict
            Standard linear model; see Notes for details.

        Notes
        -----
        The returned slm will have the following fields:

        - slm['X'] : (numpy.array) the design matrix.
        - slm['df'] (int) the degrees of freedom.
        - slm['SSE'] : (numpy.array) sum of square errors.
        - slm['coef'] : (numpy.array) coefficients of the linear model.
        - slm['V'] : (numpy.array) Variance matrix bases, only included for mixed effects models.
        - slm['k'] : (numpy.array) Number of variates.
        - slm['r'] : (numpy.array) Coefficients of the first (q-1) components of 'V' divided by their sum. Coefficients are clamped to a minimum of 0.01 * standard deviation.
        - slm['dr'] : (numpy.array) Increments of 'r' = 0.1 * sd.
        - slm['resl'] : (numpy.array) Sum over observations of squares of differences of normalized residuals along each edge. Only returned if `surf is not None`.
        - slm['tri'] : (numpy.array) Triangle indices of the surface. Only return when `surf is not None`.
        - slm['lat'] : (numpy.array) Neighbors in the input lattice.

        """

        if isinstance(Y, Term):
            Y = Y.m.to_numpy()

        n, v = Y.shape[:2]  # number of samples x number of points
        k = 1 if Y.ndim == 2 else Y.shape[2]  # number of features

        # Get data from term/random
        V = None
        if isinstance(self._model, Random):
            self.X, Vl = self._model.mean.matrix.values, self._model.variance.matrix.values

            # check in var contains intercept (constant term)
            _, q = Vl.shape
            II = np.identity(n).ravel()

            r = II - Vl @ (la.pinv(Vl) @ II)
            if (r ** 2).mean() > np.finfo(float).eps:
                warnings.warn("Did you forget an error term, I? :-)")

            if q > 1 or q == 1 and np.abs(II - Vl.T).sum() > 0:
                V = Vl.reshape(n, n, -1)

        else:  # No random term
            q = 1
            if isinstance(self._model, Term):
                self.X = self._model.matrix.values
            else:
                if self._model.size > 1:
                    warnings.warn(
                        "If you don"
                        "t convert vectors to terms you can "
                        "get unexpected results :-("
                    )
                self.X = self._model

            if self.X.shape[0] == 1:
                self.X = np.tile(self.X, (n, 1))

        # check if term (x) contains intercept (constant term)
        pinvX = la.pinv(self.X)
        r = 1 - self.X @ pinvX.sum(1)
        if (r ** 2).mean() > np.finfo(float).eps:
            warnings.warn("Did you forget an error term, I? :-)")

        p = self.X.shape[1]  # number of predictors
        self.df = n - la.matrix_rank(self.X)  # degrees of freedom

        if k == 1:  # Univariate

            if q == 1:  # Fixed effects

                if V is None:  # OLS
                    coef = pinvX @ Y
                    Y = Y - self.X @ coef

                else:
                    V = V / np.diag(V).mean(0)
                    Vmh = la.inv(la.cholesky(V).T)

                    coef = (la.pinv(Vmh @ self.X) @ Vmh) @ Y
                    Y = Vmh @ Y - (Vmh @ self.X) @ coef

                sse = np.sum(Y ** 2, axis=0)

            else:  # mixed effects

                q1 = q - 1

                V /= np.diagonal(V, axis1=0, axis2=1).mean(-1)
                slm_r = np.zeros((q1, v))

                # start Fisher scoring algorithm
                R = np.eye(n) - self.X @ la.pinv(self.X)
                RVV = (V.T @ R.T).T
                E = Y.T @ (R.T @ RVV.T)
                E *= Y.T
                E = E.sum(-1)

                RVV2 = np.zeros([n, n, q])
                E2 = np.zeros([q, v])
                for j in range(q):
                    RV2 = R @ V[..., j]
                    E2[j] = (Y * ((RV2 @ R) @ Y)).sum(0)
                    RVV2[..., j] = RV2

                M = np.einsum("ijk,jil->kl", RVV, RVV, optimize="optimal")

                theta = la.pinv(M) @ E
                tlim = np.sqrt(2 * np.diag(la.pinv(M))) * self._thetalim
                tlim = tlim[:, None] * theta.sum(0)
                m = theta < tlim
                theta[m] = tlim[m]
                r = theta[:q1] / theta.sum(0)

                Vt = 2 * la.pinv(M)
                m1 = np.diag(Vt)
                m2 = 2 * Vt.sum(0)
                Vr = m1[:q1] - m2[:q1] * slm_r.mean(1) + Vt.sum() * (r ** 2).mean(-1)
                dr = np.sqrt(Vr) * self._drlim

                # Extra Fisher scoring iterations
                for it in range(self._niter):
                    irs = np.round(r.T / dr)
                    ur, jr = np.unique(irs, axis=0, return_inverse=True)
                    nr = ur.shape[0]
                    for ir in range(nr):
                        iv = jr == ir
                        rv = r[:, iv].mean(1)

                        Vs = (1 - rv.sum()) * V[..., q - 1]
                        Vs += (V[..., :q1] * rv).sum(-1)

                        Vinv = la.inv(Vs)
                        VinvX = Vinv @ self.X
                        G = la.pinv(self.X.T @ VinvX) @ VinvX.T
                        R = Vinv - VinvX @ G

                        RVV = (V.T @ R.T).T
                        E = Y[:, iv].T @ (R.T @ RVV.T)
                        E *= Y[:, iv].T
                        E = E.sum(-1)

                        M = np.einsum("ijk,jil->kl", RVV, RVV, optimize="optimal")

                        thetav = la.pinv(M) @ E
                        tlim = np.sqrt(2 * np.diag(la.pinv(M))) * self._thetalim
                        tlim = tlim[:, None] * thetav.sum(0)

                        m = thetav < tlim
                        thetav[m] = tlim[m]
                        theta[:, iv] = thetav

                    r = theta[:q1] / theta.sum(0)

                # finish Fisher scoring
                irs = np.round(r.T / dr)
                ur, jr = np.unique(irs, axis=0, return_inverse=True)
                nr = ur.shape[0]

                coef = np.zeros((p, v))
                sse = np.zeros(v)
                for ir in range(nr):
                    iv = jr == ir
                    rv = r[:, iv].mean(1)

                    Vs = (1 - rv.sum()) * V[..., q - 1]
                    Vs += (V[..., :q1] * rv).sum(-1)

                    # Vmh = la.inv(la.cholesky(Vs).T)
                    Vmh = la.inv(la.cholesky(Vs))
                    VmhX = Vmh @ self.X
                    G = (la.pinv(VmhX.T @ VmhX) @ VmhX.T) @ Vmh

                    coef[:, iv] = G @ Y[:, iv]
                    R = Vmh - VmhX @ G
                    Y[:, iv] = R @ Y[:, iv]
                    sse[iv] = (Y[:, iv] ** 2).sum(0)

                self.r = r
                self.dr = dr[:,None]
            self.SSE = sse[None]

        else:  # multivariate
            if q > 1:
                raise ValueError(
                    "Multivariate mixed effects models not yet " "implemented :-("
                )

            if V is None:
                X2 = self.X
            else:
                V = V / np.diag(V).mean(0)
                Vmh = la.inv(la.cholesky(V)).T
                X2 = Vmh @ self.X
                pinvX = la.pinv(X2)
                Y = Vmh @ Y

            coef = pinvX @ Y.T.swapaxes(-1, -2)
            Y = Y - (X2 @ coef).swapaxes(-1, -2).T
            coef = coef.swapaxes(-1, -2).T

            k2 = k * (k + 1) // 2
            sse = np.zeros((k2, v))
            j = -1
            for j1 in range(k):
                for j2 in range(j1 + 1):
                    j = j + 1
                    sse[j] = (Y[..., j1] * Y[..., j2]).sum(0)

        self.coef = coef
        self.SSE = sse
        self.V = V

        if self._surf is not None and (
            isinstance(self._surf, BSPolyData) or ("tri" in self._surf or "lat" in surf)
        ):
            if isinstance(self._surf, BSPolyData):
                self.tri = np.array(get_cells(self._surf)) + 1
            else:
                if 'tri' in self._surf:
                    self.tri = self._surf['tri']
                else:
                    self.lat = self._surf['lat']

            edges = mesh_edges(self._surf)

            n_edges = edges.shape[0]

            resl = np.zeros((n_edges, k))
            Y = np.atleast_3d(Y)

            for j in range(k):
                normr = np.sqrt(sse[((j + 1) * (j + 2) // 2) - 1])
                for i in range(n):
                    u = Y[i, :, j] / normr
                    resl[:, j] += np.diff(u[edges], axis=1).ravel() ** 2
            self.resl = resl

    def t_test(self):
        """T statistics for a contrast in a univariate or multivariate model.

        Parameters
        ----------
        slm : dict
            Standard linear model, see Notes for details.
        contrast : numpy.array
            Vector containing the contrast in observations.

        Returns
        -------
        slm : dict
            Standard linear model, see Notes for details.

        Notes
        -----
        The input model should be the output model of linear_model. It should
        contain the following fields:

        - slm['X'] : (numpy.array) the design matrix.
        - slm['df'] (int) the degrees of freedom.
        - slm['SSE'] : (numpy.array) sum of square errors.
        - slm['coef'] : (numpy.array) coefficients of the linear model.
        - slm['V'] : (numpy.array) Variance matrix bases, only included for mixed effects models.
        - slm['k'] : (numpy.array) Number of variates.
        - slm['r'] : (numpy.array) Coefficients of the first (q-1) components of 'V' divided by their sum. Coefficients are clamped to a minimum of 0.01 * standard deviation.
        - slm['dr'] : (numpy.array) Increments of 'r' = 0.1 * sd.
        - slm['resl'] : (numpy.array) Sum over observations of squares of differences of normalized residuals along each edge. Only returned if `surf is not None`.
        - slm['tri'] : (numpy.array) Triangle indices of the surface. Only return when `surf is not None`.
        - slm['lat'] : (numpy.array) Neighbors in the input lattice.

        The output model will add the following fields.
        - slm['c'] : (numpy.array), contrasts in coefficents of the linear model.
        - slm['k'] : (int) number of variates
        - slm['ef'] : (numpy.array) array of effects.
        - slm['sd'] : (numpy.array) standard deviations of the effects.
        - slm['t'] : (numpy.array) T-values computed with a t-test (univariate) or Hotelling T^2 (multivariate).
        - slm['dfs'] : (numpy.array) effective degrees of freedom. Absent if q==1.

        Note that the contrast in the observations is used to determine the intended
        contrast in the model coefficients, slm.c. However there is some ambiguity
        in this when the model contains redundant terms. An example of such a model
        is 1 + Gender (Gender by itself does not contain redundant terms). Only one
        of the ambiguous contrasts is estimable (i.e. has slm.sd < Inf), and this is
        the one chosen, though it may not be the contrast that you intended. To
        check this, compare the contrast in the coefficients slm.c to the actual
        design matrix in slm.X. Note that the redundant columns of the design matrix
        have weights given by the rows of null(slm.X,'r')'
        """

        if isinstance(self._contrast, Term):
            self._contrast = self._contrast.m.to_numpy()

        def null(A, eps=1e-15):
            u, s, vh = scipy.linalg.svd(A)
            null_mask = s <= eps
            null_space = scipy.compress(null_mask, vh, axis=0)
            return scipy.transpose(null_space)

        if not isinstance(self.df, np.ndarray):
            self.df = np.array([self.df])

        if self._contrast.ndim == 1:
            self._contrast = np.reshape(self._contrast, (-1, 1))

        [n, p] = np.shape(self.X)
        pinvX = np.linalg.pinv(self.X)

        if len(self._contrast) <= p:
            c = np.concatenate(
                (self._contrast, np.zeros((1, p - np.shape(self._contrast)[1]))), axis=1
            ).T

            if np.square(np.dot(null_space(self.X).T, c)).sum() / np.square(
                c
            ).sum() > np.spacing(1):
                sys.exit("Contrast is not estimable :-(")

        else:
            c = np.dot(pinvX, self._contrast)
            r = self._contrast - np.dot(self.X, c)

            if np.square(np.ravel(r, "F")).sum() / np.square(
                np.ravel(self._contrast, "F")
            ).sum() > np.spacing(1):
                warnings.warn("Contrast is not in the model :-( ")

        self.c = c.T
        self.df = self.df[len(self.df) - 1]

        if np.ndim(self.coef) == 2:
            k = 1
            self.k = k

            if self.r is None:
                # fixed effect
                if self.V is not None:
                    Vmh = np.linalg.inv(cholesky(self.V).T)
                    pinvX = np.linalg.pinv(np.dot(Vmh, self.X))
                Vc = np.sum(np.square(np.dot(c.T, pinvX)), axis=1)
            else:
                # mixed effect
                q1, v = np.shape(self.r)
                q = q1 + 1
                nc = np.shape(self.dr)[1]
                chunk = math.ceil(v / nc)
                irs = np.zeros((q1, v))

                for ic in range(1, nc + 1):
                    v1 = 1 + (ic - 1) * chunk
                    v2 = np.min((v1 + chunk - 1, v))

                    vc = v2 - v1 + 1

                    irs[:, int(v1 - 1) : int(v2)] = np.around(
                        np.multiply(
                            self.r[:, int(v1 - 1) : int(v2)],
                            np.tile(1 / self.dr[:, (ic - 1)], (1, vc)),
                        )
                    )

                ur, ir, jr = np.unique(irs, axis=0, return_index=True, return_inverse=True)
                ir = ir + 1
                jr = jr + 1
                nr = np.shape(ur)[0]
                self.dfs = np.zeros((1, v))
                Vc = np.zeros((1, v))

                for ir in range(1, nr + 1):
                    iv = (jr == ir).astype(int)
                    rv = self.r[:, (iv - 1)].mean(axis=1)
                    V = (1 - rv.sum()) * self.V[:, :, (q - 1)]

                    for j in range(1, q1 + 1):
                        V = V + rv[(j - 1)] * self.V[:, :, (j - 1)]

                    Vinv = np.linalg.inv(V)
                    VinvX = np.dot(Vinv, self.X)
                    Vbeta = np.linalg.pinv(np.dot(self.X.T, VinvX))
                    G = np.dot(Vbeta, VinvX.T)
                    Gc = np.dot(G.T, c)
                    R = Vinv - np.dot(VinvX, G)
                    E = np.zeros((q, 1))
                    RVV = np.zeros((np.shape(self.V)))
                    M = np.zeros((q, q))

                    for j in range(1, q + 1):
                        E[(j - 1)] = np.dot(Gc.T, np.dot(self.V[:, :, (j - 1)], Gc))
                        RVV[:, :, (j - 1)] = np.dot(R, self.V[:, :, (j - 1)])

                    for j1 in range(1, q + 1):
                        for j2 in range(j1, q + 1):
                            M[(j1 - 1), (j2 - 1)] = (
                                RVV[:, :, (j1 - 1)] * RVV[:, :, (j2 - 1)].T
                            ).sum()
                            M[(j2 - 1), (j1 - 1)] = M[(j1 - 1), (j2 - 1)]

                    vc = np.dot(c.T, np.dot(Vbeta, c))
                    iv = (jr == ir).astype(int)
                    Vc[iv - 1] = vc
                    self.dfs[iv - 1] = np.square(vc) / np.dot(
                        E.T, np.dot(np.linalg.pinv(M), E)
                    )

            self.ef = np.dot(c.T, self.coef)
            self.sd = np.sqrt(np.multiply(Vc, self.SSE) / self.df)
            self.t = np.multiply(
                np.divide(self.ef, (self.sd + (self.sd <= 0))), self.sd > 0
            )

        else:
            # multivariate
            p, v, k = np.shape(self.coef)
            self.k = k
            self.ef = np.zeros((k, v))

            for j in range(0, k):
                self.ef[j, :] = np.dot(c.T, coef[:, :, j])

            j = np.arange(1, k + 1)
            jj = (np.multiply(j, j + 1) / 2) - 1
            jj = jj.astype(int)

            vf = np.divide(np.sum(np.square(np.dot(c.T, pinvX)), axis=1), self.df)
            self.sd = np.sqrt(vf * self.SSE[jj, :])

            if k == 2:
                det = np.multiply(self.SSE[0, :], self.SSE[2, :]) - np.square(
                    self.SSE[1, :]
                )

                self.t = (
                    np.multiply(np.square(self.ef[0, :]), self.SSE[2, :])
                    + np.multiply(np.square(self.ef[1, :]), self.SSE[0, :])
                    - np.multiply(
                        np.multiply(2 * self.ef[0, :], self.ef[1, :]), self.SSE[1, :]
                    )
                )

            if k == 3:
                det = (
                    np.multiply(
                        self.SSE[0, :],
                        (
                            np.multiply(self.SSE[2, :], self.SSE[5, :])
                            - np.square(self.SSE[4, :])
                        ),
                    )
                    - np.multiply(self.SSE[5, :], np.square(self.SSE[1, :]))
                    + np.multiply(
                        self.SSE[3, :],
                        (
                            np.multiply(self.SSE[1, :], self.SSE[4, :]) * 2
                            - np.multiply(self.SSE[2, :], self.SSE[3, :])
                        ),
                    )
                )

                self.t = np.multiply(
                    np.square(self.ef[0, :]),
                    (
                        np.multiply(self.SSE[2, :], self.SSE[5, :])
                        - np.square(self.SSE[4, :])
                    ),
                )

                self.t = self.t + np.multiply(
                    np.square(self.ef[1, :]),
                    (
                        np.multiply(self.SSE[0, :], self.SSE[5, :])
                        - np.square(self.SSE[3, :])
                    ),
                )

                self.t = self.t + np.multiply(
                    np.square(self.ef[2, :]),
                    (
                        np.multiply(self.SSE[0, :], self.SSE[2, :])
                        - np.square(self.SSE[1, :])
                    ),
                )

                self.t = self.t + np.multiply(
                    2 * self.ef[0, :],
                    np.multiply(
                        self.ef[1, :],
                        (
                            np.multiply(self.SSE[3, :], self.SSE[4, :])
                            - np.multiply(self.SSE[1, :], self.SSE[5, :])
                        ),
                    ),
                )

                self.t = self.t + np.multiply(
                    2 * self.ef[0, :],
                    np.multiply(
                        self.ef[2, :],
                        (
                            np.multiply(self.SSE[1, :], self.SSE[4, :])
                            - np.multiply(self.SSE[2, :], self.SSE[3, :])
                        ),
                    ),
                )

                self.t = self.t + np.multiply(
                    2 * self.ef[1, :],
                    np.multiply(
                        self.ef[2, :],
                        (
                            np.multiply(self.SSE[1, :], self.SSE[3, :])
                            - np.multiply(self.SSE[0, :], self.SSE[4, :])
                        ),
                    ),
                )

            if k > 3:
                sys.exit("Hotelling" "s T for k>3 not programmed yet")

            self.t = np.multiply(np.divide(self.t, (det + (det <= 0))), (det > 0)) / vf
            self.t = np.multiply(np.sqrt(self.t + (self.t <= 0)), (self.t > 0))
