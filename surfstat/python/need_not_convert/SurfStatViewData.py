"""
python version of SurfStatViewData
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np


@deprecated("BrainSpace dependency")
def py_SurfStatViewData(data, surf, title, background):
    """Basic viewer for surface data.
    Usage: [ a, cb ] = SurfStatViewData( data, surf [,title [,background]] )

    Parameters
    ----------
    data :
        1 x v vector of data, v=#vertices
    surf.coord :
        3 x v matrix of coordinates.
    surf.tri :
        t x 3 matrix of triangle indices, 1-based, t=#triangles.
    title :
        any string, data name by default.
    background :
        background colour, any matlab ColorSpec, such as
        'white' (default), 'black'=='k', 'r'==[1 0 0], [1 0.4 0.6] (pink) etc.
        Letter and line colours are inverted if background is dark (mean<0.5).

    Returns
    -------
    a :
        vector of handles to the axes, left to right, top to bottom.
    cb :
        handle to the colorbar.
    """

    sys.exit("Function py_SurfStatViewData is not implemented yet")
