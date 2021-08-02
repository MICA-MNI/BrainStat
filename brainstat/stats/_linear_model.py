import warnings
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import numpy.linalg as la
from brainspace.mesh.mesh_elements import get_cells
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat.mesh.utils import mesh_edges
from brainstat.stats.terms import FixedEffect, MixedEffect


def _linear_model(self, Y: Union[np.ndarray, FixedEffect]) -> None:
    """Fits linear mixed effects models to surface data and estimates resels.

    Parameters
    ----------
    self : brainstat.stats.SLM.SLM
        Initialized SLM object.
    Y : numpy array, brainstat.stats.terms.FixedEffect
        Input data of shape (samples, vertices, features).

    """

    # PATCHWORK FIX: Y is modified in place a lot which also modifies it outside
    # the scope of this function. Lets just make a deep copy.
    # TODO: Stop modifying in place.
    Y_copy = deepcopy(Y)

    if isinstance(Y_copy, FixedEffect):
        Y_copy = Y_copy.m.to_numpy()
    Y_copy = np.atleast_3d(Y_copy)

    n_samples = Y.shape[0]

    self.X, self.V = _get_design_matrix(self, n_samples)
    _check_constant_term(self.X)

    self.df = n_samples - la.matrix_rank(self.X)
    residuals, self.V, self.coef, self.SSE, self.r, self.dr = _run_linear_model(
        self, Y_copy
    )

    if self.surf is not None:
        self.resl, mesh_connections = _compute_resls(self, residuals)
        key = list(mesh_connections.keys())[0]
        setattr(self, key, mesh_connections[key])


def _run_linear_model(
    self, Y: np.ndarray
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Runs a linear model and returns relevant parameters.

    Parameters
    ----------
    Y : numpy.array
        Response variable matrix

    Returns
    -------
    numpy.array
        Model residuals.
    numpy.array
        Variance matrix bases.
    numpy.array
        Model coefficients.
    numpy.array
        Sum of squared errors.
    numpy.array, None
        Matrix of coefficients of the components of slm.V divided by their sum.
        None is returned for fixed effects models.
    numpy.array, None
        Vector of increments in slm.r None is returned for fixed effects models.

    Raises
    ------
    ValueError
        Returns an error if a multivariate mixed effects model is requested.
    """
    n_random_effects = _get_n_random_effects(self)
    r = None
    dr = None

    if Y.shape[2] == 1:  # Univariate
        Y = Y[:, :, 0]

        if n_random_effects == 1:  # Fixed effects
            residuals, V, coef, SSE = _model_univariate_fixed_effects(self, Y)
        else:  # mixed effects
            residuals, V, coef, SSE, r, dr = _model_univariate_mixed_effects(self, Y)

    else:  # multivariate
        if n_random_effects > 1:
            raise ValueError("Multivariate mixed effects models not implemented.")
        residuals, V, coef, SSE = _model_multivariate_fixed_effects(self, Y)
    return residuals, V, coef, SSE, r, dr


def _model_univariate_fixed_effects(
    self, Y: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Runs a univariate fixed effects linear model

    Parameters
    ----------
    Y : numpy.array
        Response variable matrix

    Returns
    -------
    numpy.array
        Model residuals.
    numpy.array
        Variance matrix bases.
    numpy.array
        Model coefficients.
    numpy.array
        Sum of squared errors.
    """
    if self.V is None:  # OLS
        coef = la.pinv(self.X) @ Y
        residuals = Y - self.X @ coef
        V = None

    else:
        V = self.V / np.diag(self.V).mean(0)
        Vmh = la.inv(la.cholesky(V).T)

        coef = (la.pinv(Vmh @ self.X) @ Vmh) @ Y
        residuals = Vmh @ Y - (Vmh @ self.X) @ coef

    SSE = np.sum(residuals ** 2, axis=0)
    SSE = SSE[None]

    return residuals, V, coef, SSE


def _model_univariate_mixed_effects(self, Y: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Runs a univariate linear mixed effects model.

    Parameters
    ----------
    Y : numpy.array
        Response variable matrix

    Returns
    -------
    numpy.array
        Model residuals.
    numpy.array
        Variance matrix bases.
    numpy.array
        Model coefficients.
    numpy.array
        Sum of squared errors.
    numpy.array
        Matrix of coefficients of the components of slm.V divided by their sum.
    numpy.array
        Vector of increments in slm.r
    """
    n_samples, n_vertices = Y.shape[:2]
    n_random_effects = _get_n_random_effects(self)
    n_predictors = self.X.shape[1]

    q1 = n_random_effects - 1

    V = self.V / np.diagonal(self.V, axis1=0, axis2=1).mean(-1)
    slm_r = np.zeros((q1, n_vertices))

    # start Fisher scoring algorithm
    R = np.eye(n_samples) - self.X @ la.pinv(self.X)
    RVV = (V.T @ R.T).T
    E = Y.T @ (R.T @ RVV.T)
    E *= Y.T
    E = E.sum(-1)

    RVV2 = np.zeros([n_samples, n_samples, n_random_effects])
    E2 = np.zeros([n_random_effects, n_vertices])
    for j in range(n_random_effects):
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
    for _ in range(self.niter):
        irs = np.round(r.T / dr)
        ur, jr = np.unique(irs, axis=0, return_inverse=True)
        nr = ur.shape[0]
        for ir in range(nr):
            iv = jr == ir
            rv = r[:, iv].mean(1)

            Vs = (1 - rv.sum()) * V[..., n_random_effects - 1]
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

    coef = np.zeros((n_predictors, n_vertices))
    SSE = np.zeros(n_vertices)
    residuals = Y
    for ir in range(nr):
        iv = jr == ir
        rv = r[:, iv].mean(1)

        Vs = (1 - rv.sum()) * V[..., n_random_effects - 1]
        Vs += (V[..., :q1] * rv).sum(-1)

        # Vmh = la.inv(la.cholesky(Vs).T)
        Vmh = la.inv(la.cholesky(Vs))
        VmhX = Vmh @ self.X
        G = (la.pinv(VmhX.T @ VmhX) @ VmhX.T) @ Vmh

        coef[:, iv] = G @ residuals[:, iv]
        R = Vmh - VmhX @ G
        residuals[:, iv] = R @ residuals[:, iv]
        SSE[iv] = (residuals[:, iv] ** 2).sum(0)
    SSE = SSE[None]
    return residuals, V, coef, SSE, r, dr[:, None]


def _model_multivariate_fixed_effects(
    self, Y: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Runs a multivariate linear fixed effects model.

    Parameters
    ----------
    Y : numpy.array
        Response variable matrix

    Returns
    -------
    numpy.array
        Model residuals.
    numpy.array
        Variance matrix bases.
    numpy.array
        Model coefficients.
    numpy.array
        Sum of squared errors.
    """
    k = Y.shape[2]
    n_vertices = Y.shape[1]

    if self.V is None:
        X2 = self.X
        V = self.V
        pinvX = la.pinv(self.X)
    else:
        V = self.V / np.diag(self.V).mean(0)
        Vmh = la.inv(la.cholesky(V)).T
        X2 = Vmh @ self.X
        pinvX = la.pinv(X2)
        Y = Vmh @ Y

    coef = pinvX @ Y.T.swapaxes(-1, -2)
    residuals = Y - (X2 @ coef).swapaxes(-1, -2).T
    coef = coef.swapaxes(-1, -2).T

    k2 = k * (k + 1) // 2
    SSE = np.zeros((k2, n_vertices))
    j = -1
    for j1 in range(k):
        for j2 in range(j1 + 1):
            j = j + 1
            SSE[j] = (residuals[..., j1] * residuals[..., j2]).sum(0)
    return residuals, V, coef, SSE


def _get_design_matrix(self, n_samples: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Wrapper for fetching the design matrix

    Parameters
    ----------
    n_samples: int
        Scalar describing the number of samples.

    Returns
    -------
    numpy.array
        Design matrix
    numpy.array, None
        Variance matrix bases. Returns None for fixed effects models.

    """
    if isinstance(self.model, MixedEffect):
        X, V = _get_mixed_design(self)
    else:
        X = _get_fixed_design(self, n_samples)
        V = None
    return X, V


def _get_mixed_design(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fetches the design matrix from a mixed effects model.

    Parameters
    ----------
    n_samples : numpy.array
        Scalar describing the number of samples.

    Returns
    -------
    numpy.array
        Design matrix
    numpy.array
        Variance matrix bases.
    """
    X = self.model.mean.matrix.values
    n_samples = X.shape[0]
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
        V = random_effects.reshape(n_samples, n_samples, -1)
    else:
        V = None

    return X, V


def _get_fixed_design(self, n_samples: int) -> np.ndarray:
    """Fetches the design matrix from a fixed effects model.

    Parameters
    ----------
    n_samples : int
        Scalar describing the number of samples.

    Returns
    -------
    numpy.array
        Design matrix
    """
    if isinstance(self.model, FixedEffect):
        X = self.model.matrix.values
    else:
        if self.model.size > 1:
            warnings.warn(
                "If you don"
                "'t convert vectors to terms you can "
                "get unexpected results :-("
            )
        X = self.model

    if X.shape[0] == 1:
        X = np.tile(X, (n_samples, 1))
    return X


def _compute_resls(self, Y: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Computes the sum over observations of squares of differences of
    normalized residuals along each edge.

    Parameters
    ----------
    Y : numpy.array
        Response variable residual matrix.

    Returns
    -------
    numpy.array
        Sum over observations of squares of differences of normalized residuals
        along each edge.
    dict
        Dictionary containing the mesh connections in either triangle or lattice
        format. The dictionary's sole key is 'tri' for triangle connections or
        'lat' for lattice connections.
    """
    if isinstance(self.surf, BSPolyData):
        mesh_connections = {"tri": np.array(get_cells(self.surf)) + 1}
    else:
        key = "tri" if "tri" in self.surf else "lat"
        mesh_connections = {key: self.surf[key]}

    edges = mesh_edges(self.surf, self.mask)

    n_edges = edges.shape[0]

    Y = np.atleast_3d(Y)
    resl = np.zeros((n_edges, Y.shape[2]))

    for j in range(Y.shape[2]):
        normr = np.sqrt(self.SSE[((j + 1) * (j + 2) // 2) - 1])
        for i in range(Y.shape[0]):
            u = Y[i, :, j] / normr
            resl[:, j] += np.diff(u[edges], axis=1).ravel() ** 2

    return resl, mesh_connections


def _check_constant_term(X: np.ndarray) -> None:
    """Checks whether an error term was provided.

    Parameters
    ----------
    X : numpy.array
        Design matrix.
    """
    r = 1 - X @ la.pinv(X).sum(1)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn("Did you forget an error term, I? :-)")


def _get_n_random_effects(self) -> int:
    """Gets the number of random effects.

    Returns
    -------
    int
        Number of random effects.
    """
    if isinstance(self.model, MixedEffect):
        n_random_effects = self.model.variance.matrix.values.shape[1]
    else:
        n_random_effects = 1
    return n_random_effects
