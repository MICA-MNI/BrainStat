"""Creation, fitting, and comparison of linear models."""
import warnings
import numpy as np
import numpy.linalg as la
import scipy
from scipy.linalg import null_space
from scipy.linalg import cholesky
from cmath import sqrt
from ..mesh.utils import mesh_edges
from .terms import Term, Random
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainspace.mesh.mesh_elements import get_cells


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

    if "r" in slm1.keys() or "r" in slm2.keys():
        warnings.warn("Mixed effects models not programmed yet.")

    if slm1["df"] > slm2["df"]:
        X1 = slm1["X"]
        X2 = slm2["X"]
        df1 = slm1["df"]
        df2 = slm2["df"]
        SSE1 = slm1["SSE"]
        SSE2 = slm2["SSE"]
        slm = slm2.copy()
    else:
        X1 = slm2["X"]
        X2 = slm1["X"]
        df1 = slm2["df"]
        df2 = slm1["df"]
        SSE1 = slm2["SSE"]
        SSE2 = slm1["SSE"]
        slm = slm1.copy()

    r = X1 - np.dot(np.dot(X2, np.linalg.pinv(X2)), X1)
    d = np.sum(r.flatten() ** 2) / np.sum(X1.flatten() ** 2)

    if d > np.spacing(1):
        print("Models are not nested.")
        return

    slm["df"] = np.array([[df1 - df2, df2]])
    h = SSE1 - SSE2

    # if slm['coef'] is 3D and third dimension is 1, then squeeze it to 2D
    if np.ndim(slm["coef"]) == 3 and np.shape(slm["coef"])[2] == 1:
        x1, x2, x3 = np.shape(slm["coef"])
        slm["coef"] = slm["coef"].reshape(x1, x2)

    if np.ndim(slm["coef"]) == 2:
        slm["k"] = np.array(1)
        slm["t"] = np.dot(h / (SSE2 + (SSE2 <= 0)) * (SSE2 > 0), df2 / (df1 - df2))
    elif np.ndim(slm["coef"]) > 2:
        k2, v = np.shape(SSE2)
        k = np.around((np.sqrt(1 + 8 * k2) - 1) / 2)
        slm["k"] = np.array(k)
        if k > 3:
            print("Roy" "s max root for k>3 not programmed yet.")
            return

        l = min(k, df1 - df2)
        slm["t"] = np.zeros((int(l), int(v)))

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
            slm["t"][0, :] = (a1 + s1) * d
            if l == 2:
                slm["t"][1, :] = (a1 - s1) * d
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
                slm["t"][j, :] = z[j, :] * d
    return slm


def linear_model(Y, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):
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
            warnings.warn("Did you forget an error term, I? :-)")

        if q > 1 or q == 1 and np.abs(II - Vl.T).sum() > 0:
            V = Vl.reshape(n, n, -1)

    else:  # No random term
        q = 1
        if isinstance(M, Term):
            X = M.matrix.values
        else:
            if M.size > 1:
                warnings.warn(
                    "If you don"
                    "t convert vectors to terms you can "
                    "get unexpected results :-("
                )
            X = M

        if X.shape[0] == 1:
            X = np.tile(X, (n, 1))

    # check if term (x) contains intercept (constant term)
    pinvX = la.pinv(X)
    r = 1 - X @ pinvX.sum(1)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn("Did you forget an error term, I? :-)")

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
            tlim = np.sqrt(2 * np.diag(la.pinv(M))) * thetalim
            tlim = tlim[:, None] * theta.sum(0)
            m = theta < tlim
            theta[m] = tlim[m]
            r = theta[:q1] / theta.sum(0)

            Vt = 2 * la.pinv(M)
            m1 = np.diag(Vt)
            m2 = 2 * Vt.sum(0)
            Vr = m1[:q1] - m2[:q1] * slm_r.mean(1) + Vt.sum() * (r ** 2).mean(-1)
            dr = np.sqrt(Vr) * drlim

            # Extra Fisher scoring iterations
            for it in range(niter):
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
                    tlim = np.sqrt(2 * np.diag(la.pinv(M))) * thetalim
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

            slm.update(dict(r=r, dr=dr[:, None]))

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

    slm.update(dict(coef=coef, SSE=sse))
    if V is not None:
        slm["V"] = V

    if surf is not None and (
        isinstance(surf, BSPolyData) or ("tri" in surf or "lat" in surf)
    ):
        if isinstance(surf, BSPolyData):
            slm["tri"] = np.array(get_cells(surf)) + 1
        else:
            key = "tri" if "tri" in surf else "lat"
            slm[key] = surf[key]

        edges = mesh_edges(surf)

        n_edges = edges.shape[0]

        resl = np.zeros((n_edges, k))
        Y = np.atleast_3d(Y)

        for j in range(k):
            normr = np.sqrt(sse[((j + 1) * (j + 2) // 2) - 1])
            for i in range(n):
                u = Y[i, :, j] / normr
                resl[:, j] += np.diff(u[edges], axis=1).ravel() ** 2
        slm["resl"] = resl

    return slm


def t_test(slm, contrast):
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

    def null(A, eps=1e-15):
        u, s, vh = scipy.linalg.svd(A)
        null_mask = s <= eps
        null_space = scipy.compress(null_mask, vh, axis=0)
        return scipy.transpose(null_space)

    if not isinstance(slm["df"], np.ndarray):
        slm["df"] = np.array([slm["df"]])

    if contrast.ndim == 1:
        contrast = np.reshape(contrast, (-1, 1))

    [n, p] = np.shape(slm["X"])
    pinvX = np.linalg.pinv(slm["X"])

    if len(contrast) <= p:
        c = np.concatenate(
            (contrast, np.zeros((1, p - np.shape(contrast)[1]))), axis=1
        ).T

        if np.square(np.dot(null_space(slm["X"]).T, c)).sum() / np.square(
            c
        ).sum() > np.spacing(1):
            sys.exit("Contrast is not estimable :-(")

    else:
        c = np.dot(pinvX, contrast)
        r = contrast - np.dot(slm["X"], c)

        if np.square(np.ravel(r, "F")).sum() / np.square(
            np.ravel(contrast, "F")
        ).sum() > np.spacing(1):
            warnings.warn("Contrast is not in the model :-( ")

    slm["c"] = c.T
    slm["df"] = slm["df"][len(slm["df"]) - 1]

    if np.ndim(slm["coef"]) == 2:
        k = 1
        slm["k"] = k

        if not "r" in slm.keys():
            # fixed effect
            if "V" in slm.keys():
                Vmh = np.linalg.inv(cholesky(slm["V"]).T)
                pinvX = np.linalg.pinv(np.dot(Vmh, slm["X"]))

            Vc = np.sum(np.square(np.dot(c.T, pinvX)), axis=1)
        else:
            # mixed effect
            q1, v = np.shape(slm["r"])
            q = q1 + 1
            nc = np.shape(slm["dr"])[1]
            chunck = np.ceil(v / nc)
            irs = np.zeros((q1, v))

            for ic in range(1, nc + 1):
                v1 = 1 + (ic - 1) * chunck
                v2 = np.min((v1 + chunck - 1, v))
                vc = v2 - v1 + 1

                irs[:, int(v1 - 1) : int(v2)] = np.around(
                    np.multiply(
                        slm["r"][:, int(v1 - 1) : int(v2)],
                        np.tile(1 / slm["dr"][:, (ic - 1)], (1, vc)),
                    )
                )

            ur, ir, jr = np.unique(irs, axis=0, return_index=True, return_inverse=True)
            ir = ir + 1
            jr = jr + 1
            nr = np.shape(ur)[0]
            slm["dfs"] = np.zeros((1, v))
            Vc = np.zeros((1, v))

            for ir in range(1, nr + 1):
                iv = (jr == ir).astype(int)
                rv = slm["r"][:, (iv - 1)].mean(axis=1)
                V = (1 - rv.sum()) * slm["V"][:, :, (q - 1)]

                for j in range(1, q1 + 1):
                    V = V + rv[(j - 1)] * slm["V"][:, :, (j - 1)]

                Vinv = np.linalg.inv(V)
                VinvX = np.dot(Vinv, slm["X"])
                Vbeta = np.linalg.pinv(np.dot(slm["X"].T, VinvX))
                G = np.dot(Vbeta, VinvX.T)
                Gc = np.dot(G.T, c)
                R = Vinv - np.dot(VinvX, G)
                E = np.zeros((q, 1))
                RVV = np.zeros((np.shape(slm["V"])))
                M = np.zeros((q, q))

                for j in range(1, q + 1):
                    E[(j - 1)] = np.dot(Gc.T, np.dot(slm["V"][:, :, (j - 1)], Gc))
                    RVV[:, :, (j - 1)] = np.dot(R, slm["V"][:, :, (j - 1)])

                for j1 in range(1, q + 1):
                    for j2 in range(j1, q + 1):
                        M[(j1 - 1), (j2 - 1)] = (
                            RVV[:, :, (j1 - 1)] * RVV[:, :, (j2 - 1)].T
                        ).sum()
                        M[(j2 - 1), (j1 - 1)] = M[(j1 - 1), (j2 - 1)]

                vc = np.dot(c.T, np.dot(Vbeta, c))
                iv = (jr == ir).astype(int)
                Vc[iv - 1] = vc
                slm["dfs"][iv - 1] = np.square(vc) / np.dot(
                    E.T, np.dot(np.linalg.pinv(M), E)
                )

        slm["ef"] = np.dot(c.T, slm["coef"])
        slm["sd"] = np.sqrt(np.multiply(Vc, slm["SSE"]) / slm["df"])
        slm["t"] = np.multiply(
            np.divide(slm["ef"], (slm["sd"] + (slm["sd"] <= 0))), slm["sd"] > 0
        )

    else:
        # multivariate
        p, v, k = np.shape(slm["coef"])
        slm["k"] = k
        slm["ef"] = np.zeros((k, v))

        for j in range(0, k):
            slm["ef"][j, :] = np.dot(c.T, slm["coef"][:, :, j])

        j = np.arange(1, k + 1)
        jj = (np.multiply(j, j + 1) / 2) - 1
        jj = jj.astype(int)

        vf = np.divide(np.sum(np.square(np.dot(c.T, pinvX)), axis=1), slm["df"])
        slm["sd"] = np.sqrt(vf * slm["SSE"][jj, :])

        if k == 2:
            det = np.multiply(slm["SSE"][0, :], slm["SSE"][2, :]) - np.square(
                slm["SSE"][1, :]
            )

            slm["t"] = (
                np.multiply(np.square(slm["ef"][0, :]), slm["SSE"][2, :])
                + np.multiply(np.square(slm["ef"][1, :]), slm["SSE"][0, :])
                - np.multiply(
                    np.multiply(2 * slm["ef"][0, :], slm["ef"][1, :]), slm["SSE"][1, :]
                )
            )

        if k == 3:
            det = (
                np.multiply(
                    slm["SSE"][0, :],
                    (
                        np.multiply(slm["SSE"][2, :], slm["SSE"][5, :])
                        - np.square(slm["SSE"][4, :])
                    ),
                )
                - np.multiply(slm["SSE"][5, :], np.square(slm["SSE"][1, :]))
                + np.multiply(
                    slm["SSE"][3, :],
                    (
                        np.multiply(slm["SSE"][1, :], slm["SSE"][4, :]) * 2
                        - np.multiply(slm["SSE"][2, :], slm["SSE"][3, :])
                    ),
                )
            )

            slm["t"] = np.multiply(
                np.square(slm["ef"][0, :]),
                (
                    np.multiply(slm["SSE"][2, :], slm["SSE"][5, :])
                    - np.square(slm["SSE"][4, :])
                ),
            )

            slm["t"] = slm["t"] + np.multiply(
                np.square(slm["ef"][1, :]),
                (
                    np.multiply(slm["SSE"][0, :], slm["SSE"][5, :])
                    - np.square(slm["SSE"][3, :])
                ),
            )

            slm["t"] = slm["t"] + np.multiply(
                np.square(slm["ef"][2, :]),
                (
                    np.multiply(slm["SSE"][0, :], slm["SSE"][2, :])
                    - np.square(slm["SSE"][1, :])
                ),
            )

            slm["t"] = slm["t"] + np.multiply(
                2 * slm["ef"][0, :],
                np.multiply(
                    slm["ef"][1, :],
                    (
                        np.multiply(slm["SSE"][3, :], slm["SSE"][4, :])
                        - np.multiply(slm["SSE"][1, :], slm["SSE"][5, :])
                    ),
                ),
            )

            slm["t"] = slm["t"] + np.multiply(
                2 * slm["ef"][0, :],
                np.multiply(
                    slm["ef"][2, :],
                    (
                        np.multiply(slm["SSE"][1, :], slm["SSE"][4, :])
                        - np.multiply(slm["SSE"][2, :], slm["SSE"][3, :])
                    ),
                ),
            )

            slm["t"] = slm["t"] + np.multiply(
                2 * slm["ef"][1, :],
                np.multiply(
                    slm["ef"][2, :],
                    (
                        np.multiply(slm["SSE"][1, :], slm["SSE"][3, :])
                        - np.multiply(slm["SSE"][0, :], slm["SSE"][4, :])
                    ),
                ),
            )

        if k > 3:
            sys.exit("Hotelling" "s T for k>3 not programmed yet")

        slm["t"] = np.multiply(np.divide(slm["t"], (det + (det <= 0))), (det > 0)) / vf
        slm["t"] = np.multiply(np.sqrt(slm["t"] + (slm["t"] <= 0)), (slm["t"] > 0))

    return slm
