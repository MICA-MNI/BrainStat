from sklearn.base import BaseEstimator


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
        self.cluster_threshold=cluster_threshold
        self.mask=None

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
    #TODO: Add onetailed/twotailed
