"""
python version of SufStatROILabel
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np

def py_SurfStatROILabel(lhlabel, rhlabel, nl, nr):
    """ROI from a FreeSurfer .label file.

    Parameters
    ----------
    lhlabel :
        FreeSurfer .label file for the left  hemisphere, or empty [].
    rhlabel :
        FreeSurfer .label file for the right hemisphere, or empty [].
    nl :
        number of vertices in the left  hemisphere, 163842 by default.
    nr :
        number of vertices in the right hemisphere, 163842 by default.

    Returns
    -------
    ROI : 1D ndarray logical | int, shape = (1, n_vertices)
        1 x (nl+nr) logical vector, 1=labelled point, 0=otherwise.
    """
    sys.exit("Function py_SurfStatROILabel is under development")
