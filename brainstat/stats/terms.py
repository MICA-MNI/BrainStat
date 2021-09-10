"""Classes for fixed, mixed, and random effects."""
import re
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from brainstat._typing import ArrayLike
from brainstat._utils import deprecated


def check_names(
    x: Union[int, "FixedEffect", pd.DataFrame, pd.Series]
) -> Optional[List[str]]:
    """Return True if `x` is FixedEffect, Series or DataFrame."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.columns.tolist()
    elif isinstance(x, FixedEffect):
        return x.names
    else:
        return None


def to_df(
    x: Union[int, ArrayLike, "FixedEffect"],
    n: int = 1,
    names: Optional[Union[str, Sequence[str]]] = None,
    idx: Optional[int] = None,
) -> pd.DataFrame:
    """Convert input to DataFrame.

    Parameters
    ----------
    x : int, array-like FixedEffect
        Input data.
    n : int, optional
        If input is a scalar, broadcast to column of `n` entries.
        Default is 1.
    names : str, sequence of str or None, optional
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
    if isinstance(x, pd.DataFrame):
        df = x
    elif isinstance(x, FixedEffect):
        df = x.m
    elif isinstance(x, pd.Series):
        df = pd.DataFrame(x)
    else:
        x_2d = np.atleast_2d(x)
        if x_2d.shape[0] == 1:
            x_2d = x_2d.T
        df = pd.DataFrame(x_2d)
        has_names = False

    if df.empty:
        return df

    if df.size == 1 and n > 1:
        df = pd.concat([df] * n, ignore_index=True)

    if names is None and (idx is not None or not has_names):
        if idx is None:
            idx = 0
        names = ["x%d" % i for i in range(idx, idx + df.shape[1])]

    if names is not None:
        if len(names) != df.shape[1]:
            raise ValueError(
                f"Number of columns {df.shape[1]} does not "
                f"coincide with column names {names}"
            )
        df.columns = names

    return pd.get_dummies(df)


def get_index(df: pd.DataFrame) -> Optional[int]:
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


def check_duplicate_names(
    df1: pd.DataFrame, df2: Optional[pd.DataFrame] = None
) -> None:
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
        raise ValueError(f"Variables must have different names: {names}")


def remove_identical_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with duplicate names across dataframes.

    Parameters
    ----------
    df1 : DataFrame
        Input dataframe from which to drop columns.
    df2 : DataFrame
        Reference dataFrame

    Returns
    -------
    DataFrame
        df1 with columns appearing in df2 removed.

    Raises
    ------
    ValueError
        If there are columns with duplicate names but no duplicate values.

    """
    names = np.intersect1d(df1.columns, df2.columns)
    for col in names:
        if np.array_equal(df1[col], df2[col]):
            df1 = df1.drop(col, axis=1)
        elif (df1[col].size == 1 or df2[col].size == 1) and np.all(
            df1[col].to_numpy() == df2[col].to_numpy()
        ):
            # Assume its an intercept term.
            df1 = df1.drop(col, axis=1)
        else:
            raise ValueError(
                f"Column {col} must be identical for duplicate removal. Either alter the column name or remove the duplicate data."
            )
    return df1


def remove_duplicate_columns(df: pd.DataFrame, tol: float = 1e-8) -> List[str]:
    """Remove duplicate columns.

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
    # keep = df.round().T.drop_duplicates(keep="last").T.columns  # Slow!!
    idx = np.unique(df.round().values, axis=1, return_index=True)[-1]
    keep = df.columns[sorted(idx)]
    return keep


class FixedEffect(object):
    """Build a term object for a linear model.

    Parameters
    ----------
    x : array-like or DataFrame, optional
        If None, the term is empty. Default is None.
    names : str or list of str, optional
        Names for each column in `x`. If None, it defauts to {'x0', 'x1', ...}.
        Default is None.
    add_intercept : bool, optional
        If true, adds an intercept term. Defaults to True.

    Attributes
    ----------
    x : DataFrame
        Design matrix.
    names : list of str
        Names of columns in the design matrix.

    See Also
    --------
    MixedEffect: MixedEffect term

    Examples
    --------
    >>> t = FixedEffect()
    >>> t.is_empty
    True

    >>> t1 = FixedEffect(np.arange(5), names='t1')
    >>> t2 = FixedEffect(np.random.randn(5, 1), names=['t2'])
    >>> t3 = t1 + t2
    >>> t3.shape
    (5, 3)

    """

    tolerance = 1e-8

    def __init__(
        self,
        x: Optional[Union[ArrayLike, pd.DataFrame]] = None,
        names: Optional[Union[str, Sequence[str]]] = None,
        add_intercept: bool = True,
    ) -> None:

        if x is None:
            self.m = pd.DataFrame()
            return

        if isinstance(x, FixedEffect):
            self.m = x.m
            return

        if np.isscalar(x) and names is None:
            names = ["intercept"]

        if isinstance(names, str):
            names = [names]

        self.m = to_df(x, names=names).reset_index(drop=True)
        if add_intercept and "intercept" not in self.names:
            self.m.insert(0, "intercept", 1)
        check_duplicate_names(self.m)

    def _broadcast(
        self, t: Union[ArrayLike, "FixedEffect"], idx: Optional[int] = None
    ) -> pd.DataFrame:
        df = to_df(t, idx=idx)
        if self.shape[0] > 1 and df.shape[0] == 1:
            df = to_df(df, n=self.shape[0])
        elif self.shape[0] == 1 and df.shape[0] > 1:
            self.m = to_df(self.m, n=df.shape[0])
        elif not df.empty and self.shape[0] != df.shape[0]:
            raise ValueError(f"Cannot broadcast shape {df.shape} to " f"{self.shape}.")
        return df

    def _add(
        self, t: Union[ArrayLike, "FixedEffect", "MixedEffect"], side: str = "left"
    ) -> "FixedEffect":
        if isinstance(t, MixedEffect):
            return NotImplemented

        if self.empty:
            return FixedEffect(t, add_intercept=False)

        idx = None
        if check_names(t) is None:
            idx = get_index(self.m)

        df = self._broadcast(t, idx=idx)
        if df.empty:
            return self

        df = remove_identical_columns(df, self.m)

        # check_duplicate_names(df, df2=self.m)
        terms = [self.m, df]
        names = [self.names, list(df.columns)]
        if side == "right":
            terms = terms[::-1]
            names = names[::-1]
        df = pd.concat(terms, axis=1)
        df.columns = names[0] + names[1]
        cols = remove_duplicate_columns(df, tol=self.tolerance)
        return FixedEffect(df[cols], add_intercept=False)

    def __add__(
        self, t: Union[ArrayLike, "FixedEffect", "MixedEffect"]
    ) -> "FixedEffect":
        return self._add(t)

    def __radd__(
        self, t: Union[ArrayLike, "FixedEffect", "MixedEffect"]
    ) -> "FixedEffect":
        return self._add(t, side="right")

    def __sub__(self, t: Union[ArrayLike, "FixedEffect"]) -> "FixedEffect":
        if self.empty:
            return self
        df = self._broadcast(t)
        if df.empty:
            return self
        df /= df.abs().sum(0)
        df.index = self.m.index

        m = self.m / self.m.abs().sum(0)
        merged = m.T.merge(df.T, how="outer", indicator=True)
        mask = (merged._merge.values == "left_only")[: self.m.shape[1]]
        return FixedEffect(self.m[self.m.columns[mask]], add_intercept=False)

    def _mul(
        self, t: Union[ArrayLike, "FixedEffect", "MixedEffect"], side: str = "left"
    ) -> "FixedEffect":
        if isinstance(t, MixedEffect):
            return NotImplemented

        if self.is_empty:
            return self
        if np.isscalar(t):
            if t == 1:
                return self
            m = self.m * t
            if side == "right":
                names = [f"{t}*{k}" for k in self.names]
            else:
                names = [f"{k}*{t}" for k in self.names]
            return FixedEffect(m, names=names, add_intercept=False)

        df = self._broadcast(t)
        if df.empty:
            return FixedEffect()
        prod = []
        names = []
        for c in df.columns:
            prod.append(df[[c]].values * self.m)
            if side == "left":
                names.extend([f"{k}*{c}" for k in self.names])
            else:
                names.extend([f"{c}*{k}" for k in self.names])

        df = pd.concat(prod, axis=1)
        df.columns = names
        cols = remove_duplicate_columns(df, tol=self.tolerance)
        return FixedEffect(df[cols], add_intercept=False)

    def __mul__(
        self, t: Union[ArrayLike, "FixedEffect", "MixedEffect"]
    ) -> "FixedEffect":
        return self._mul(t)

    def __rmul__(
        self, t: Union[ArrayLike, "FixedEffect", "MixedEffect"]
    ) -> "FixedEffect":
        return self._mul(t, side="right")

    def __pow__(self, p: int) -> "FixedEffect":
        if p > 1:
            return self * self ** (p - 1)
        return self

    def __repr__(self) -> str:
        return self.m.__repr__()

    def _repr_html_(self) -> str:
        return self.m._repr_html_()

    @property
    def is_scalar(self) -> bool:
        return self.size == 1

    @property
    def matrix(self) -> pd.DataFrame:
        return self.m

    @property
    def names(self) -> List[str]:
        return object.__getattribute__(self, "m").columns.tolist()

    @property
    def is_empty(self) -> bool:
        return self.m.empty

    def __getattr__(self, name: str) -> Any:
        if name in object.__getattribute__(self, "names"):
            return object.__getattribute__(self, "m")[name].values
        if name in {"shape", "size", "empty"}:
            return getattr(self.m, name)
        return object.__getattribute__(self, name)


class MixedEffect:
    """Build a random term object for a linear model.

    Parameters
    ----------
    ran : array-like or DataFrame, optional
        For the random effects. If None, the random term is empty.
        Default is None.
    fix : array-like or DataFrame, optional
        If None, the fixed effects.
    name_ran : str, list, optional
        Name(s) for the random term(s). If None, it defauts to 'xi'.
        Default is None.
    name_fix : str, optional
        Name for the `fix` term. If None, it defauts to 'xi'.
        Default is None.
    ranisvar : bool, optional
        If True, `ran` is already a term for the variance. Default is False.

    Attributes
    ----------
    mean : FixedEffect
        FixedEffect for the mean.

    variance : FixedEffect
        FixedEffect for the variance.

    See Also
    --------
    FixedEffect: FixedEffect object

    Examples
    --------
    >>> r = MixedEffect()
    >>> r.is_empty
    True

    >>> r2 = MixedEffect(np.arange(5), name_ran='r1')
    >>> r2.mean.is_empty
    True
    >>> r2.variance.shape
    (25, 2)

    """

    def __init__(
        self,
        ran: Optional[Union[ArrayLike, pd.DataFrame]] = None,
        fix: Optional[Union[ArrayLike, pd.DataFrame]] = None,
        name_ran: Optional[Union[str, Sequence[str]]] = None,
        name_fix: Optional[Union[str, Sequence[str]]] = None,
        ranisvar: bool = False,
        add_intercept: bool = True,
        add_identity: bool = True,
    ) -> None:

        if isinstance(ran, MixedEffect):
            self.mean = ran.mean  # type: ignore
            self.variance = ran.variance  # type: ignore
            return

        if ran is None:
            self.variance = FixedEffect()
        else:
            ran = to_df(ran)
            if not ranisvar:
                if ran.size == 1:
                    name_ran = "I"
                    v = ran.values.flat[0]
                    if v != 1:
                        name_ran += f"{v}**2"
                else:
                    name = check_names(ran)
                    if name is not None and name_ran is None:
                        name_ran = name
                ran = ran @ ran.T
                ran = ran.values.ravel()

            self.variance = FixedEffect(ran, names=name_ran, add_intercept=False)
        self.mean = FixedEffect(fix, names=name_fix, add_intercept=add_intercept)

        if add_identity:
            I = MixedEffect(1, name_ran="I", add_identity=False)
            tmp_mixed = self + I
            self.variance = tmp_mixed.variance

        self.set_identity_last()

    def set_identity_last(self) -> None:
        """Sets the identity matrix column last.

        Raises
        ------
        ValueError
            Raised if "I" occurs more than once in the names.
        """

        if self.variance.is_empty:
            return

        if self.variance.m.size == 1:
            # Class is the identity.
            return

        n = int(round(np.sqrt(self.variance.m.shape[0])))
        I = np.reshape(np.identity(n), (-1, 1))
        index = np.argwhere(np.all(self.variance.m.to_numpy() == I, axis=0))

        if index.size == 0:
            return
        elif index.size == 1:
            names = self.variance.names
            names.append(names.pop(index[0][0]))
            self.variance.m = self.variance.m[names]
        else:
            raise ValueError("Found the identity matrix twice in the dataframe.")

    def broadcast_to(self, r1: "MixedEffect", r2: "MixedEffect") -> FixedEffect:
        if r1.variance.shape[0] == 1:
            v = np.eye(max(r2.shape[0], int(np.sqrt(r2.shape[2]))))
            return FixedEffect(v.ravel(), names="I", add_intercept=False)
        return r1.variance

    def _add(
        self, r: Union[FixedEffect, "MixedEffect"], side: str = "left"
    ) -> "MixedEffect":
        if not isinstance(r, MixedEffect):
            r = MixedEffect(fix=r, add_intercept=False, add_identity=False)

        r.variance = self.broadcast_to(r, self)
        self.variance = self.broadcast_to(self, r)
        if side == "left":
            ran = self.variance + r.variance
            fix = self.mean + r.mean
        else:
            ran = r.variance + self.variance
            fix = r.mean + self.mean

        s = MixedEffect(
            ran=ran, fix=fix, ranisvar=True, add_intercept=False, add_identity=False
        )
        s.set_identity_last()
        return s

    def __add__(self, r: Union[FixedEffect, "MixedEffect"]) -> "MixedEffect":  # type: ignore[misc]
        return self._add(r)

    def __radd__(self, r: Union[FixedEffect, "MixedEffect"]) -> "MixedEffect":  # type: ignore[misc]
        return self._add(r, side="right")

    def _sub(
        self, r: Union[FixedEffect, "MixedEffect"], side: str = "left"
    ) -> "MixedEffect":
        if not isinstance(r, MixedEffect):
            r = MixedEffect(fix=r, add_intercept=False, add_identity=False)
        r.variance = self.broadcast_to(r, self)
        self.variance = self.broadcast_to(self, r)
        if side == "left":
            ran = self.variance - r.variance
            fix = self.mean - r.mean
        else:
            ran = r.variance - self.variance
            fix = r.mean - self.mean
        return MixedEffect(
            ran=ran, fix=fix, ranisvar=True, add_intercept=False, add_identity=False
        )

    def __sub__(self, r: Union[FixedEffect, "MixedEffect"]) -> "MixedEffect":
        return self._sub(r)

    def __rsub__(self, r: Union[FixedEffect, "MixedEffect"]) -> "MixedEffect":
        return self._sub(r, side="right")

    def _mul(
        self, r: Union[FixedEffect, "MixedEffect"], side: str = "left"
    ) -> "MixedEffect":
        if not isinstance(r, MixedEffect):
            r = MixedEffect(fix=r, add_intercept=False, add_identity=False)
        r.variance = self.broadcast_to(r, self)
        self.variance = self.broadcast_to(self, r)

        if side == "left":
            ran = self.variance * r.variance
            fix = self.mean * r.mean
        else:
            ran = r.variance * self.variance
            fix = r.mean * self.mean
        s = MixedEffect(
            ran=ran, fix=fix, ranisvar=True, add_intercept=False, add_identity=False
        )

        if self.mean.matrix.values.size > 0:
            x = self.mean.matrix.values.T / self.mean.matrix.abs().values.max()
            t = FixedEffect()
            for i in range(x.shape[0]):
                for j in range(i + 1):
                    if i == j:
                        t = t + FixedEffect(
                            np.outer(x[i], x[j]).T.ravel(),
                            names=self.mean.names[i],
                            add_intercept=False,
                        )
                    else:
                        xs = x[i] + x[j]
                        xs_name = f"({self.mean.names[i]}+{self.mean.names[j]})"
                        xd = x[i] - x[j]
                        xd_name = f"({self.mean.names[i]}-{self.mean.names[j]})"

                        v = np.outer(xs, xs) / 4
                        t = t + FixedEffect(
                            v.ravel(), names=xs_name, add_intercept=False
                        )
                        v = np.outer(xd, xd) / 4
                        t = t + FixedEffect(
                            v.ravel(), names=xd_name, add_intercept=False
                        )

            s.variance = s.variance + t * r.variance

        if r.mean.matrix.values.size > 0:
            x = r.mean.matrix.values.T / r.mean.matrix.abs().values.max()
            t = FixedEffect()
            for i in range(x.shape[0]):
                for j in range(i + 1):
                    if i == j:
                        t = t + FixedEffect(
                            np.outer(x[i], x[j]).ravel(), names=r.mean.names[i]
                        )
                    else:
                        xs = x[i] + x[j]
                        xs_name = f"({r.mean.names[i]}+{r.mean.names[j]})"
                        xd = x[i] - x[j]
                        xd_name = f"({r.mean.names[i]}-{r.mean.names[j]})"

                        v = np.outer(xs, xs) / 4
                        t = t + FixedEffect(
                            v.ravel(), names=xs_name, add_intercept=False
                        )
                        v = np.outer(xd, xd) / 4
                        t = t + FixedEffect(
                            v.ravel(), names=xd_name, add_intercept=False
                        )
            s.variance = s.variance + self.variance * t
        s.set_identity_last()
        return s

    def __mul__(self, r: Union[FixedEffect, "MixedEffect"]) -> "MixedEffect":  # type: ignore[misc]
        return self._mul(r)

    def __rmul__(self, r: Union[FixedEffect, "MixedEffect"]) -> "MixedEffect":  # type: ignore[misc]
        return self._mul(r, side="right")

    def __pow__(self, p: int) -> "MixedEffect":
        if p > 1:
            return self * self ** (p - 1)
        return self

    @property
    def empty(self) -> bool:
        return self.mean.empty and self.variance.empty

    @property
    def shape(self) -> np.ndarray:
        return self.mean.shape + self.variance.shape

    def _repr_html_(self) -> str:
        return (
            "Mean:\n"
            + self.mean._repr_html_()
            + "\n\nVariance:\n"
            + self.variance._repr_html_()
        )


## Deprecated functions
@deprecated("Please use FixedEffect instead.")
def Term(x=None, names=None):
    return FixedEffect(x=x, names=names, add_intercept=False)


@deprecated("Please use MixedEffect instead.")
def Random(ran=None, fix=None, name_ran=None, name_fix=None, ranisvar=False):
    return MixedEffect(
        ran=ran,
        fix=fix,
        name_ran=name_ran,
        name_fix=name_fix,
        ranisvar=ranisvar,
        add_intercept=False,
        add_identity=False,
    )
