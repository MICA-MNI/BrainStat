import math
import sys
import warnings

import numpy as np
from scipy.linalg import cholesky, null_space

from .terms import FixedEffect


def _t_test(self) -> None:
    """T statistics for a contrast in a univariate or multivariate model.

    Parameters
    ----------
    self : brainstat.stats.SLM.SLM
        SLM object that has already run linear_model


    """

    if isinstance(self.contrast, FixedEffect):
        self.contrast = self.contrast.m.to_numpy()

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

            ur, _, jr = np.unique(irs.T, axis=0, return_index=True, return_inverse=True)
            jr = jr + 1
            nr = np.shape(ur)[0]
            self.dfs = np.zeros((1, v))
            Vc = np.zeros((1, v))

            for ir in range(1, nr + 1):
                iv = (jr == ir).astype(bool)
                rv = self.r[:, iv].mean(axis=1)
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
                Vc[0, iv] = vc
                self.dfs[0, iv] = np.square(vc) / np.dot(
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

        # Initialize some variables
        self.t = np.zeros(v)
        indices_lower = np.tril_indices(self.k)
        sse_indices = np.zeros((self.k, self.k), dtype=int)
        sse_indices[indices_lower] = np.arange(0, self.k * (self.k + 1) / 2)
        sse_indices += np.tril(sse_indices, -1).T

        M = np.zeros((self.k + 1, self.k + 1))
        M_ef = np.zeros((self.k + 1, self.k + 1), dtype=bool)
        M_ef[1:, 0] = True
        M_ef[0, 1:] = True
        M_sse = np.zeros((self.k + 1, self.k + 1), dtype=bool)
        M_sse[1:, 1:] = True

        ef_duplicate = np.concatenate((self.ef, self.ef), axis=0)
        for i in range(v):
            sse_vertex = self.SSE[:, i]
            sse_matrix = sse_vertex[sse_indices]

            det_sse = np.linalg.det(sse_matrix)
            if det_sse <= 0:
                self.t[i] = 0
            else:
                M[M_ef] = ef_duplicate[:, i]
                M[M_sse] = sse_matrix.flatten()

                self.t[i] = np.sqrt(-np.linalg.det(M) / det_sse / vf)

    self.t = np.atleast_2d(self.t)
