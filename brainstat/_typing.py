import sys
from typing import Any

""" Python 3.6/3.7 compatibility for the ArrayLike type hint. 
In these versions we simply accept anything. The tests in 3.8+
should catch all errors anyway. """
if sys.version_info[1] >= 8:
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any
