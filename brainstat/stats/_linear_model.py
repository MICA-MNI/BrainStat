import warnings
import numpy as np
import numpy.linalg as la
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
    Y : numpy array, brainstat.stats.terms.Term
        Input data of shape (samples, vertices, features).

    """

    if isinstance(Y, Term):
        Y = Y.m.to_numpy()
    Y = np.atleast_3d(Y)
    n_samples = Y.shape[0]

    _set_design_matrix(self, n_samples)
    _check_error_term(self.X)

    self.df = n_samples - la.matrix_rank(self.X)
    residuals = _run_linear_model(self, Y)
    if self.surf is not None:
        _compute_resls(self, residuals)


def _run_linear_model(self, Y):

    k = Y.shape[2]
    n_random_effects = _get_n_random_effects(self)

    if k == 1:  # Univariate
        Y = Y[:, :, 0]

        if n_random_effects == 1:  # Fixed effects
            residuals = _model_univariate_fixed_effects(self, Y)
        else:  # mixed effects
            residuals = _model_univariate_mixed_effects(self, Y)

    else:  # multivariate
        if n_random_effects > 1:
            raise ValueError(
                "Multivariate mixed effects models not yet " "implemented :-("
            )
        residuals = _model_multivariate_fixed_effects(self, Y)
    return residuals


def _model_univariate_fixed_effects(self, Y):

    if self.V is None:  # OLS
        self.coef = la.pinv(self.X) @ Y
        residuals = Y - self.X @ self.coef

    else:
        self.V = self.V / np.diag(self.V).mean(0)
        Vmh = la.inv(la.cholesky(self.V).T)

        self.coef = (la.pinv(Vmh @ self.X) @ Vmh) @ Y
        residuals = Vmh @ Y - (Vmh @ self.X) @ self.coef

    self.SSE = np.sum(residuals ** 2, axis=0)
    self.SSE = self.SSE[None]

    return residuals


def _model_univariate_mixed_effects(self, Y):

    n_samples, n_vertices = Y.shape[:2]
    n_random_effects = _get_n_random_effects(self)
    n_predictors = self.X.shape[1]

    q1 = n_random_effects - 1

    self.V /= np.diagonal(self.V, axis1=0, axis2=1).mean(-1)
    slm_r = np.zeros((q1, n_vertices))

    # start Fisher scoring algorithm
    R = np.eye(n_samples) - self.X @ la.pinv(self.X)
    RVV = (self.V.T @ R.T).T
    E = Y.T @ (R.T @ RVV.T)
    E *= Y.T
    E = E.sum(-1)

    RVV2 = np.zeros([n_samples, n_samples, n_random_effects])
    E2 = np.zeros([n_random_effects, n_vertices])
    for j in range(n_random_effects):
        RV2 = R @ self.V[..., j]
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
    for _ in range(self.niter):
        irs = np.round(r.T / dr)
        ur, jr = np.unique(irs, axis=0, return_inverse=True)
        nr = ur.shape[0]
        for ir in range(nr):
            iv = jr == ir
            rv = r[:, iv].mean(1)

            Vs = (1 - rv.sum()) * self.V[..., n_random_effects - 1]
            Vs += (self.V[..., :q1] * rv).sum(-1)

            Vinv = la.inv(Vs)
            VinvX = Vinv @ self.X
            G = la.pinv(self.XX.T @ VinvX) @ VinvX.T
            R = Vinv - VinvX @ G

            RVV = (self.V.T @ R.T).T
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

    self.coef = np.zeros((n_predictors, n_vertices))
    self.SSE = np.zeros(n_vertices)
    for ir in range(nr):
        iv = jr == ir
        rv = r[:, iv].mean(1)

        Vs = (1 - rv.sum()) * self.V[..., n_random_effects - 1]
        Vs += (self.V[..., :q1] * rv).sum(-1)

        # Vmh = la.inv(la.cholesky(Vs).T)
        Vmh = la.inv(la.cholesky(Vs))
        VmhX = Vmh @ self.X
        G = (la.pinv(VmhX.T @ VmhX) @ VmhX.T) @ Vmh

        self.coef[:, iv] = G @ Y[:, iv]
        R = Vmh - VmhX @ G
        Y[:, iv] = R @ Y[:, iv]
        self.SSE[iv] = (Y[:, iv] ** 2).sum(0)
    self.SSE = self.SSE[None]

    self.r = r
    self.dr = dr[:, None]


def _model_multivariate_fixed_effects(self, Y):

    k = Y.shape[2]
    n_vertices = Y.shape[1]

    if self.V is None:
        X2 = self.X
        pinvX = la.pinv(self.X)
    else:
        self.V = self.V / np.diag(self.V).mean(0)
        Vmh = la.inv(la.cholesky(self.V)).T
        X2 = Vmh @ self.X
        pinvX = la.pinv(X2)
        Y = Vmh @ Y

    self.coef = pinvX @ Y.T.swapaxes(-1, -2)
    Y = Y - (X2 @ self.coef).swapaxes(-1, -2).T
    self.coef = self.coef.swapaxes(-1, -2).T

    k2 = k * (k + 1) // 2
    self.SSE = np.zeros((k2, n_vertices))
    j = -1
    for j1 in range(k):
        for j2 in range(j1 + 1):
            j = j + 1
            self.SSE[j] = (Y[..., j1] * Y[..., j2]).sum(0)
    return Y


def _set_design_matrix(self, n_samples):

    if isinstance(self.model, Random):
        _set_mixed_design(self)
    else:
        _set_fixed_design(self, n_samples)


def _get_n_random_effects(self):
    if isinstance(self.model, Random):
        _, n_random_effects = self.model.variance.matrix.values.shape[1]
    else:
        n_random_effects = 1
    return n_random_effects


def _set_mixed_design(self):
    self.X = self.model.mean.matrix.values
    n_samples = self.X.shape[0]
    random_effects = self.model.variance.matrix.values

    # check in var contains intercept (constant term)
    _, n_random_effects = random_effects.shape
    identity = np.identity(n_samples).ravel()

    r = identity - random_effects @ (la.pinv(random_effects) @ identity)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn("Did you forget an error term, I? :-)")

    if (
        n_random_effects > 1
        or n_random_effects == 1
        and np.abs(identity - random_effects.T).sum() > 0
    ):
        self.V = random_effects.reshape(n_samples, n_samples, -1)


def _set_fixed_design(self, n_samples):

    if isinstance(self.model, Term):
        self.X = self.model.matrix.values
    else:
        if self.model.size > 1:
            warnings.warn(
                "If you don"
                "'t convert vectors to terms you can "
                "get unexpected results :-("
            )
        self.X = self.model

    if self.X.shape[0] == 1:
        self.X = np.tile(self.X, (n_samples, 1))


def _compute_resls(self, Y):

    if isinstance(self.surf, BSPolyData):
        self.tri = np.array(get_cells(self.surf)) + 1
    else:
        key = "tri" if "tri" in self.surf else "lat"
        setattr(self, key, self.surf[key])

    edges = mesh_edges(self.surf)

    n_edges = edges.shape[0]

    Y = np.atleast_3d(Y)
    self.resl = np.zeros((n_edges, Y.shape[2]))

    for j in range(Y.shape[2]):
        normr = np.sqrt(self.SSE[((j + 1) * (j + 2) // 2) - 1])
        for i in range(Y.shape[0]):
            u = Y[i, :, j] / normr
            self.resl[:, j] += np.diff(u[edges], axis=1).ravel() ** 2


def _check_error_term(X):
    r = 1 - X @ la.pinv(X).sum(1)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn("Did you forget an error term, I? :-)")
