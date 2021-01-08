"""Classes for fixed, mixed, and random effects."""

import re

import numpy as np
import pandas as pd


def check_names(x):
    """Return True if `x` is Term, Series or DataFrame."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.columns.tolist()
    if isinstance(x, Term):
        return x.names
    return None


def to_df(x, n=1, names=None, idx=None):
    """ Convert input to DataFrame.

    Parameters
    ----------
    x : array-like or Term
        Input data.
    n : int, optional
        If input is a scalar, broadcast to column of `n` entries.
        Default is 1.
    names : str, list of str or None, optional
        Names for each column in `x`. Default is None.
    idx : int or None, optional
        Staring index for variable names of the for `x{i}`.

    Returns
    -------
    df : DataFrame
        Input `x` wrapped in a DataFrame.

    """

    if x is None:
        return pd.DataFrame()

    has_names = True
    if isinstance(x, Term):
        x = x.m
    elif isinstance(x, pd.Series):
        x = pd.DataFrame(x)
    elif not isinstance(x, pd.DataFrame):
        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            x = x.T
        x = pd.DataFrame(x)
        has_names = False

    if x.empty:
        return x

    if x.size == 1 and n > 1:
        x = pd.concat([x] * n, ignore_index=True)

    if names is None and (idx is not None or not has_names):
        if idx is None:
            idx = 0
        names = ['x%d' % i for i in range(idx, idx + x.shape[1])]

    if names is not None:
        if len(names) != x.shape[1]:
            raise ValueError('Number of columns {} does not coincide with '
                             'column names {}'.format(x.shape[1], names))
        x.columns = names

    return pd.get_dummies(x)


def get_index(df):
    """Get index for column names of the form x{i}.

    If there are none, return 0.

    Parameters
    ----------
    df : DataFrame
        Input dataframe.

    Returns
    -------
    index : int
        Index for the next x column. If `df` is empty, return None.

    """
    if df.empty:
        return None
    r = [re.match(r"^x(\d+)$", c) for c in df.columns]
    r2 = [int(r1.groups()[0]) for r1 in r if r1 is not None]
    return 0 if len(r2) == 0 else max(r2) + 1


def check_duplicate_names(df1, df2=None):
    """Check columns with duplicate names.

    Parameters
    ----------
    df1 : DataFrame
        Input dataframe.
    df2 : DataFrame, optional
        If provided, check that dataframes do not contain columns with same
        names. Default is None.

    Raises
    ------
    ValueError
        If there are columns with duplicate names.

    """
    if df2 is None:
        names, counts = np.unique(df1.columns, return_counts=True)
        names = names[counts > 1]
    else:
        names = np.intersect1d(df1.columns, df2.columns)
    if names.size > 0:
        raise ValueError('Variables must have different names: {}'.
                         format(names))


def remove_duplicate_columns(df, tol=1e-8):
    """ Remove duplicate columns.

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    tol : float, optional
        Tolerance to assess duplicate columns. Default is 1e-8.

    Returns
    -------
    columns: list of str
        Columns to keep after removing duplicates.

    """

    df = df / df.abs().sum(0)
    df *= 1 / tol
    # keep = df.round().T.drop_duplicates(keep='last').T.columns  # Slow!!
    idx = np.unique(df.round().values, axis=1, return_index=True)[-1]
    keep = df.columns[np.flipud(idx)]
    return keep


class Term:
    """Build a term object for a linear model.

    Parameters
    ----------
    x : array-like or DataFrame, optional
        If None, the term is empty. Default is None.
    names : str or list of str, optional
        Names for each column in `x`. If None, it defauts to {'x0', 'x1', ...}.
        Default is None.

    Attributes
    ----------
    x : DataFrame
        Design matrix.
    names : list of str
        Names of columns in the design matrix.

    See Also
    --------
    Random: Random term

    Examples
    --------
    >>> t = Term()
    >>> t.is_empty
    True

    >>> t1 = Term(np.arange(5), names='t1')
    >>> t2 = Term(np.random.randn(5, 1), names=['t2'])
    >>> t3 = t1 + t2 + 1
    >>> t3.shape
    (5, 3)

    """

    tolerance = 1e-8

    def __init__(self, x=None, names=None):

        if x is None:
            self.m = pd.DataFrame()
            return

        if isinstance(x, Term):
            self.m = x.m
            return

        if isinstance(names, str):
            names = [names]

        self.m = to_df(x, names=names).reset_index(drop=True)
        check_duplicate_names(self.m)

    def _broadcast(self, t, idx=None):
        df = to_df(t, idx=idx)
        if self.shape[0] > 1 and df.shape[0] == 1:
            df = to_df(df, n=self.shape[0])
        elif self.shape[0] == 1 and df.shape[0] > 1:
            self.m = to_df(self.m, n=df.shape[0])
        elif not df.empty and self.shape[0] != df.shape[0]:
            raise ValueError('Cannot broadcast shape {} to '
                             '{}.'.format(df.shape, self.shape))
        return df

    def _add(self, t, side='right'):
        if isinstance(t, Random):
            return NotImplemented

        if self.empty:
            return Term(t)

        idx = None
        if check_names(t) is None:
            idx = get_index(self.m)

        df = self._broadcast(t, idx=idx)
        if df.empty:
            return self

        check_duplicate_names(df, df2=self.m)
        terms = [self.m, df]
        names = [self.names, list(df.columns)]
        if side == 'right':
            terms = terms[::-1]
            names = names[::-1]
        df = pd.concat(terms, axis=1)
        df.columns = names[0] + names[1]
        cols = remove_duplicate_columns(df, tol=self.tolerance)
        return Term(df[cols])

    def __add__(self, t):
        return self._add(t)

    def __radd__(self, t):
        return self._add(t, side='right')

    def __sub__(self, t):
        if self.empty:
            return self
        df = self._broadcast(t)
        if df.empty:
            return self
        df /= df.abs().sum(0)
        df.index = self.m.index

        m = self.m / self.m.abs().sum(0)
        merged = m.T.merge(df.T, how='outer', indicator=True)
        mask = (merged._merge.values == 'left_only')[:self.m.shape[1]]
        return Term(self.m[self.m.columns[mask]])

    def _mul(self, t, side='left'):
        if isinstance(t, Random):
            return NotImplemented

        if self.is_empty:
            return self
        if np.isscalar(t):
            if t == 1:
                return self
            m = self.m * t
            if side == 'right':
                names = ['{}*{}'.format(t, k) for k in self.names]
            else:
                names = ['{}*{}'.format(k, t) for k in self.names]
            return Term(m, names=names)

        df = self._broadcast(t)
        if df.empty:
            return Term()
        prod = []
        names = []
        for c in df.columns:
            prod.append(df[[c]].values * self.m)
            if side == 'left':
                names.extend(['{}*{}'.format(k, c) for k in self.names])
            else:
                names.extend(['{}*{}'.format(c, k) for k in self.names])

        df = pd.concat(prod, axis=1)
        df.columns = names
        cols = remove_duplicate_columns(df, tol=self.tolerance)
        return Term(df[cols])

    def __mul__(self, t):
        return self._mul(t)

    def __rmul__(self, t):
        return self._mul(t, side='right')

    def __pow__(self, p):
        if p > 1:
            return self * self ** (p - 1)
        return self

    def __repr__(self):
        return self.m.__repr__()

    def _repr_html_(self):
        return self.m._repr_html_()

    @property
    def is_scalar(self):
        return self.size == 1

    @property
    def matrix(self):
        return self.m

    @property
    def names(self):
        return self.m.columns.tolist()

    @property
    def is_empty(self):
        return self.m.empty

    def __getattr__(self, name):
        if name in self.names:
            return self.m[name].values
        if name in {'shape', 'size', 'empty'}:
            return getattr(self.m, name)
        return super().__getattribute__(name)


class Random:
    """Build a random term object for a linear model.

    Parameters
    ----------
    ran : array-like or DataFrame, optional
        For the random effects. If None, the random term is empty.
        Default is None.
    fix : array-like or DataFrame, optional
        If None, the fixed effects.
    name_ran : str, optional
        Name for the random term. If None, it defauts to 'xi'.
        Default is None.
    name_fix : str, optional
        Name for the `fix` term. If None, it defauts to 'xi'.
        Default is None.
    ranisvar : bool, optional
        If True, `ran` is already a term for the variance. Default is False.

    Attributes
    ----------
    mean : Term
        Term for the mean.

    variance : Term
        Term for the variance.

    See Also
    --------
    Term: Term object

    Examples
    --------
    >>> r = Random()
    >>> r.is_empty
    True

    >>> r2 = Random(np.arange(5), name_ran='r1')
    >>> r2.mean.is_empty
    True
    >>> r2.variance.shape
    (25, 1)

    """

    def __init__(self, ran=None, fix=None, name_ran=None, name_fix=None,
                 ranisvar=False):

        if isinstance(ran, Random):
            self.mean = ran.mean
            self.variance = ran.variance
            return

        if ran is None:
            self.variance = Term()
        else:
            ran = to_df(ran)
            if not ranisvar:
                if ran.size == 1:
                    name_ran = 'I'
                    v = ran.values.flat[0]
                    if v != 1:
                        name_ran += '{}**2'.format(v)
                else:
                    name = check_names(ran)
                    if name is not None and name_ran is None:
                        name_ran = name
                ran = ran @ ran.T
                ran = ran.values.ravel()

            self.variance = Term(ran, names=name_ran)
        self.mean = Term(fix, names=name_fix)

    def broadcast_to(self, r1, r2):
        if r1.variance.shape[0] == 1:
            v = np.eye(max(r2.shape[0], int(np.sqrt(r2.shape[2]))))
            return Term(v.ravel(), names='I')
        return r1.variance

    def _add(self, r, side='right'):
        if not isinstance(r, Random):
            r = Random(fix=r)

        r.variance = self.broadcast_to(r, self)
        self.variance = self.broadcast_to(self, r)
        if side == 'left':
            ran = self.variance + r.variance
            fix = self.mean + r.mean
        else:
            ran = r.variance + self.variance
            fix = r.mean + self.mean

        return Random(ran=ran, fix=fix, ranisvar=True)

    def __add__(self, r):
        return self._add(r)

    def __radd__(self, r):
        return self._add(r, side='right')

    def _sub(self, r, side='left'):
        if not isinstance(r, Random):
            r = Random(fix=r)
        r.variance = self.broadcast_to(r, self)
        self.variance = self.broadcast_to(self, r)
        if side == 'left':
            ran = self.variance - r.variance
            fix = self.mean - r.mean
        else:
            ran = r.variance - self.variance
            fix = r.mean - self.mean
        return Random(ran=ran, fix=fix, ranisvar=True)

    def __sub__(self, r):
        return self._sub(r)

    def __rsub__(self, r):
        return self._sub(r, side='right')

    def _mul(self, r, side='left'):
        if not isinstance(r, Random):
            r = Random(fix=r)
        r.variance = self.broadcast_to(r, self)
        self.variance = self.broadcast_to(self, r)

        if side == 'left':
            ran = self.variance * r.variance
            fix = self.mean * r.mean
        else:
            ran = r.variance * self.variance
            fix = r.mean * self.mean
        s = Random(ran=ran, fix=fix, ranisvar=True)

        x = self.mean.matrix.values.T / self.mean.matrix.abs().values.max()
        t = Term()
        for i in range(x.shape[0]):
            for j in range(i + 1):
                if i == j:
                    t = t + Term(np.outer(x[i], x[j]).T.ravel(),
                                 names=self.mean.names[i])
                else:
                    xs = x[i] + x[j]
                    xs_name = '({}+{})'.format(
                        *[self.mean.names[k] for k in [i, j]])
                    xd = x[i] - x[j]
                    xd_name = '({}-{})'.format(
                        *[self.mean.names[k] for k in [i, j]])

                    v = np.outer(xs, xs) / 4
                    t = t + Term(v.ravel(), names=xs_name)
                    v = np.outer(xd, xd) / 4
                    t = t + Term(v.ravel(), names=xd_name)

        s.variance = s.variance + t * r.variance

        x = r.mean.matrix.values.T / r.mean.matrix.abs().values.max()
        t = Term()
        for i in range(x.shape[0]):
            for j in range(i + 1):
                if i == j:
                    t = t + Term(np.outer(x[i], x[j]).ravel(),
                                 names=r.mean.names[i])
                else:
                    xs = x[i] + x[j]
                    xs_name = '({}+{})'.format(
                        *[r.mean.names[k] for k in [i, j]])
                    xd = x[i] - x[j]
                    xd_name = '({}-{})'.format(
                        *[r.mean.names[k] for k in [i, j]])

                    v = np.outer(xs, xs) / 4
                    t = t + Term(v.ravel(), names=xs_name)
                    v = np.outer(xd, xd) / 4
                    t = t + Term(v.ravel(), names=xd_name)
        s.variance = s.variance + self.variance * t
        return s

    def __mul__(self, r):
        return self._mul(r)

    def __rmul__(self, r):
        return self._mul(r, side='right')

    def __pow__(self, p):
        if p > 1:
            return self * self ** (p - 1)
        return self

    @property
    def empty(self):
        return self.mean.empty and self.variance.empty

    @property
    def shape(self):
        return self.mean.shape + self.variance.shape

    def _repr_html_(self):
        return 'Mean:\n' + self.mean._repr_html_() + '\n\nVariance:\n' + \
               self.variance._repr_html_()
