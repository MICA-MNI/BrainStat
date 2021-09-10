"""Neuroimaging statistics toolbox."""
import sys
import warnings

__version__ = "0.2.4"

if sys.version_info[1] == 6:
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn(
        "Support for Python3.6 has been dropped. Future versions may not install on Python3.6.",
        DeprecationWarning,
    )
