import sys
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar, Union

import numpy as np

""" Python 3.6/3.7 compatibility for the ArrayLike type hint. """
if sys.version_info[1] >= 8:
    from numpy.typing import ArrayLike
else:
    _T = TypeVar("_T")
    _DType = TypeVar("_DType", bound="np.dtype[Any]")
    _DType_co = TypeVar("_DType_co", covariant=True, bound="np.dtype[Any]")

    class _SupportsArray(Generic[_DType_co]):
        ...

    _NestedSequence = Union[
        _T,
        Sequence[_T],
        Sequence[Sequence[_T]],
        Sequence[Sequence[Sequence[_T]]],
        Sequence[Sequence[Sequence[Sequence[_T]]]],
    ]
    _RecursiveSequence = Sequence[Sequence[Sequence[Sequence[Sequence[Any]]]]]
    _ArrayLike = Union[
        _NestedSequence[_SupportsArray[_DType]],
        _NestedSequence[_T],
    ]
    ArrayLike = Union[
        _RecursiveSequence,
        _ArrayLike[np.dtype, Union[bool, int, float, complex, str, bytes]],
    ]
    ArrayLike = Union[
        _RecursiveSequence,
        _ArrayLike[np.dtype, Union[bool, int, float, complex, str, bytes]],
    ]
