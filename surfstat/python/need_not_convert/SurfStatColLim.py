"""
python version of SurfStatColLim
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np

@deprecated("BrainSpace dependency")
def py_SurfStatColLim(map):
    """Sets the colour limits for SurfStatView.
    Usage: cb = SurfStatColLim( clim );
    clim = [min, max] values of data for colour limits.
    cb   = handle to new colorbar.
    """
    sys.exit("Function py_SurfStatColormap is now a BrainSpace dependency")
