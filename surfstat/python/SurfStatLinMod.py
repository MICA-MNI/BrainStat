import warnings
import numpy as np
import numpy.linalg as la
import sys
sys.path.append("python")
from term import Term, Random
from SurfStatEdg import py_SurfStatEdg
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainspace.mesh.mesh_elements import get_cells

def py_SurfStatLinMod(Y, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):
    """ Fits linear mixed effects models to surface data and estimates resels.

    Parameters
    ----------
    Y : ndarray, shape = (n_samples, n_verts) or (n_samples, n_verts, n_feats)
        Surface data.
    M : Term or Random
        Design matrix.
    surf : dict, optional
        Surface triangles (surf['tri']) or volumetric data (surf['lat']).
        If 'tri', shape = (n_edges, 2). If 'lat', then it is a boolean 3D
        array. Default is None.
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
        Dictionary with the following keys:

        - 'X' : ndarray, shape = (n_samples, n_pred)
            Design matrix.
        - 'df' : int
            Degrees of freedom.
        - 'coef' : ndarray, shape = (n_pred, n_verts)
            Model coefficients.
        - 'SSE' : ndarray, shape = (n_feat, n_verts)
            Sum of square errors.
        - 'V' : ndarray, shape = (n_samples, n_samples, n_rand)
            Variance matrix bases. Only when mixed effects.
        - 'r' : ndarray, shape = (n_rand - 1, n_verts)
            Coefficients of the first (q-1) components of 'V' divided by their
            sum. Coefficients are clamped to a minimum of 0.01 x sd.
            Only when mixed effects.
        - 'dr' : ndarray
             Vector of increments in 'r' = 0.1 x sd
        - 'resl' : ndarray, (n_edges, n_feat)
            Sum over observations of squares of differences of normalized
            residuals along each edge. Only when ``surf is not None``.
        - 'tri' : ndarray, (n_cells, 3)
            Cells in surf. Only when ``surf is not None``.
        - 'lat' : ndarray
            Neighbors in lattice.

    """

    n, v = Y.shape[:2]  # number of samples x number of points
    k = 1 if Y.ndim == 2 else Y.shape[2]  # number of features

    # Get data from term/random
    V = None
    if isinstance(M, Random):
        X, Vl = M.mean.matrix.values, M.variance.matrix.values

        # check in var contains intercept (constant term)
        n2, q = Vl.shape
        II = np.identity(n).ravel()

        r = II - Vl @ (la.pinv(Vl) @ II)
        if (r ** 2).mean() > np.finfo(float).eps:
            warnings.warn('Did you forget an error term, I? :-)')

        if q > 1 or q == 1 and np.abs(II - Vl.T).sum() > 0:
            V = Vl.reshape(n, n, -1)

    else:  # No random term
        q = 1
        if isinstance(M, Term):
            X = M.matrix.values
        else:
            if M.size > 1:
                warnings.warn('If you don''t convert vectors to terms you can '
                              'get unexpected results :-(')
            X = M

        if X.shape[0] == 1:
            X = np.tile(X, (n, 1))

    # check if term (x) contains intercept (constant term)
    pinvX = la.pinv(X)
    r = 1 - X @ pinvX.sum(1)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn('Did you forget an error term, I? :-)')

    p = X.shape[1]  # number of predictors
    df = n - la.matrix_rank(X)  # degrees of freedom

    slm = dict(df=df, X=X)

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
            E = (Y.T @ (R.T @ RVV.T))
            E *= Y.T
            E = E.sum(-1)

            RVV2 = np.zeros([n, n, q])
            E2 = np.zeros([q, v])
            for j in range(q):
                RV2 = R @ V[..., j]
                E2[j] = (Y * ((RV2 @ R) @ Y)).sum(0)
                RVV2[..., j] = RV2

            M = np.einsum('ijk,jil->kl', RVV, RVV, optimize='optimal')

            theta = la.pinv(M) @ E
            tlim = np.sqrt(2*np.diag(la.pinv(M))) * thetalim
            tlim = tlim[:, None] * theta.sum(0)
            m = theta < tlim
            theta[m] = tlim[m]
            r = theta[:q1] / theta.sum(0)

            Vt = 2*la.pinv(M)
            m1 = np.diag(Vt)
            m2 = 2 * Vt.sum(0)
            Vr = m1[:q1]-m2[:q1] * slm_r.mean(1) + Vt.sum()*(r**2).mean(-1)
            dr = np.sqrt(Vr) * drlim

            # Extra Fisher scoring iterations
            for it in range(niter):
                irs = np.round(r.T / dr)
                ur, jr = np.unique(irs, axis=0, return_inverse=True)
                nr = ur.shape[0]
                for ir in range(nr):
                    iv = jr == ir
                    rv = r[:, iv].mean(1)

                    Vs = (1-rv.sum()) * V[..., q-1]
                    Vs += (V[..., :q1] * rv).sum(-1)

                    Vinv = la.inv(Vs)
                    VinvX = Vinv @ X
                    G = la.pinv(X.T @ VinvX) @ VinvX.T
                    R = Vinv - VinvX @ G

                    RVV = (V.T @ R.T).T
                    E = (Y[:, iv].T @ (R.T @ RVV.T))
                    E *= Y[:, iv].T
                    E = E.sum(-1)

                    M = np.einsum('ijk,jil->kl', RVV, RVV, optimize='optimal')

                    thetav = la.pinv(M) @ E
                    tlim = np.sqrt(2*np.diag(la.pinv(M))) * thetalim
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
                sse[iv] = (Y[:, iv]**2).sum(0)

            slm.update(dict(r=r, dr=dr[:, None]))

        sse = sse[None]

    else:  # multivariate
        if q > 1:
            raise ValueError('Multivariate mixed effects models not yet '
                             'implemented :-(')

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
            for j2 in range(j1+1):
                j = j + 1
                sse[j] = (Y[..., j1]*Y[..., j2]).sum(0)

    slm.update(dict(coef=coef, SSE=sse))
    if V is not None:
        slm['V'] = V

    if surf is not None and (isinstance(surf,BSPolyData) or ('tri' in surf or 'lat' in surf)):
        if isinstance(surf,BSPolyData):
            slm['tri'] = np.array(get_cells(surf)) + 1
        else:
            key = 'tri' if 'tri' in surf else 'lat'
            slm[key] = surf[key]

        edges = py_SurfStatEdg(surf)

        n_edges = edges.shape[0]

        resl = np.zeros((n_edges, k))
        Y = np.atleast_3d(Y)

        for j in range(k):
            normr = np.sqrt(sse[((j+1) * (j+2) // 2) - 1])
            for i in range(n):
                u = Y[i, :, j] / normr
                resl[:, j] += np.diff(u[edges], axis=1).ravel()**2
        slm['resl'] = resl

    return slm
