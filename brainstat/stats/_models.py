import warnings
import math
import numpy as np
import numpy.linalg as la
import scipy
import sys
from scipy.linalg import null_space
from scipy.linalg import cholesky
from brainstat.mesh.utils import mesh_edges
from brainstat.stats.terms import Term, Random
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainspace.mesh.mesh_elements import get_cells
from .terms import Term, Random


def linear_model(self, Y):
    """Fits linear mixed effects models to surface data and estimates resels.

    Parameters
    ----------
    self : brainstat.stats.SLM.SLM
        Initialized SLM object.
    Y : numpy array
        Input data of shape (samples, vertices, features).

    """

    if isinstance(Y, Term):
        Y = Y.m.to_numpy()

    n, v = Y.shape[:2]  # number of samples x number of points
    k = 1 if Y.ndim == 2 else Y.shape[2]  # number of features
    if Y.ndim == 3 and k == 1:
        Y = Y[:, :, 0]

    # Get data from term/random
    V = None
    if isinstance(self.model, Random):
        X, Vl = self.model.mean.matrix.values, self.model.variance.matrix.values

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
        if isinstance(self.model, Term):
            X = self.model.matrix.values
        else:
            if self.model.size > 1:
                warnings.warn(
                    "If you don"
                    "t convert vectors to terms you can "
                    "get unexpected results :-("
                )
            X = self.model

        if X.shape[0] == 1:
            X = np.tile(X, (n, 1))

    # check if term (x) contains intercept (constant term)
    pinvX = la.pinv(X)
    r = 1 - X @ pinvX.sum(1)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn("Did you forget an error term, I? :-)")

    p = X.shape[1]  # number of predictors
    df = n - la.matrix_rank(X)  # degrees of freedom

    self.df = df
    self.X = X

    if k == 1:  # Univariate

        if q == 1:  # Fixed effects

            if V is None:  # OLS
                coef = pinvX @ Y
                Y = Y - X @ coef

            else:
                V = V / np.diag(V).mean(0)
                Vmh = la.inv(la.cholesky(V).T)

                coef = (la.pinv(Vmh @ X) @ Vmh) @ Y
                Y = Vmh @ Y - (Vmh @ X) @ coef

            sse = np.sum(Y ** 2, axis=0)

        else:  # mixed effects

            q1 = q - 1

            V /= np.diagonal(V, axis1=0, axis2=1).mean(-1)
            slm_r = np.zeros((q1, v))

            # start Fisher scoring algorithm
            R = np.eye(n) - X @ la.pinv(X)
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
            tlim = np.sqrt(2 * np.diag(la.pinv(M))) * self.thetalim
            tlim = tlim[:, None] * theta.sum(0)
            m = theta < tlim
            theta[m] = tlim[m]
            r = theta[:q1] / theta.sum(0)

            Vt = 2 * la.pinv(M)
            m1 = np.diag(Vt)
            m2 = 2 * Vt.sum(0)
            Vr = m1[:q1] - m2[:q1] * slm_r.mean(1) + Vt.sum() * (r ** 2).mean(-1)
            dr = np.sqrt(Vr) * self.drlim

            # Extra Fisher scoring iterations
            for it in range(self.niter):
                irs = np.round(r.T / dr)
                ur, jr = np.unique(irs, axis=0, return_inverse=True)
                nr = ur.shape[0]
                for ir in range(nr):
                    iv = jr == ir
                    rv = r[:, iv].mean(1)

                    Vs = (1 - rv.sum()) * V[..., q - 1]
                    Vs += (V[..., :q1] * rv).sum(-1)

                    Vinv = la.inv(Vs)
                    VinvX = Vinv @ X
                    G = la.pinv(X.T @ VinvX) @ VinvX.T
                    R = Vinv - VinvX @ G

                    RVV = (V.T @ R.T).T
                    E = Y[:, iv].T @ (R.T @ RVV.T)
                    E *= Y[:, iv].T
                    E = E.sum(-1)

                    M = np.einsum("ijk,jil->kl", RVV, RVV, optimize="optimal")

                    thetav = la.pinv(M) @ E
                    tlim = np.sqrt(2 * np.diag(la.pinv(M))) * self.thetalim
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
                VmhX = Vmh @ X
                G = (la.pinv(VmhX.T @ VmhX) @ VmhX.T) @ Vmh

                coef[:, iv] = G @ Y[:, iv]
                R = Vmh - VmhX @ G
                Y[:, iv] = R @ Y[:, iv]
                sse[iv] = (Y[:, iv] ** 2).sum(0)

            self.r = r
            self.dr = dr[:, None]

        sse = sse[None]

    else:  # multivariate
        if q > 1:
            raise ValueError(
                "Multivariate mixed effects models not yet " "implemented :-("
            )

        if V is None:
            X2 = X
        else:
            V = V / np.diag(V).mean(0)
            Vmh = la.inv(la.cholesky(V)).T
            X2 = Vmh @ X
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

    if V is not None:
        self.V = V

    if self.surf is not None and (
        isinstance(self.surf, BSPolyData) or ("tri" in self.surf or "lat" in self.surf)
    ):
        if isinstance(self.surf, BSPolyData):
            self.tri = np.array(get_cells(self.surf)) + 1
        else:
            key = "tri" if "tri" in self.surf else "lat"
            setattr(self, key, self.surf[key])

        edges = mesh_edges(self.surf)

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
    self : brainstat.stats.SLM.SLM
        SLM object that has already run linear_model


    """

    if isinstance(self.contrast, Term):
        self.contrast = self.contrast.m.to_numpy()

    def null(A, eps=1e-15):
        u, s, vh = scipy.linalg.svd(A)
        null_mask = s <= eps
        null_space = scipy.compress(null_mask, vh, axis=0)
        return scipy.transpose(null_space)

    if not isinstance(self.df, np.ndarray):
        self.df = np.array([self.df])

    if self.contrast.ndim == 1:
        self.contrast = np.reshape(self.contrast, (-1, 1))

    [n, p] = np.shape(self.X)
    pinvX = np.linalg.pinv(self.X)

    if len(self.contrast) <= p:
        c = np.concatenate(
            (self.contrast, np.zeros((1, p - np.shape(self.contrast)[1]))), axis=1
        ).T

        if np.square(np.dot(null_space(self.X).T, c)).sum() / np.square(
            c
        ).sum() > np.spacing(1):
            sys.exit("Contrast is not estimable :-(")

    else:
        c = np.dot(pinvX, self.contrast)
        r = self.contrast - np.dot(self.X, c)

        if np.square(np.ravel(r, "F")).sum() / np.square(
            np.ravel(self.contrast, "F")
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
            self.ef[j, :] = np.dot(c.T, self.coef[:, :, j])

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
    self.t = np.atleast_2d(self.t)
