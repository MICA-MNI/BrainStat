import numpy as np
import sys
sys.path.append("/data/p_02323/BrainStat/surfstat/")
import surfstat_wrap as sw


def py_SurfStatMaskCut(surf):
    """Generates a mask that excludes the inter-hemisphere cut. It
    looks in -50<y<50 and -20<z<40, and mask vertices where
    |x|>thresh, where thresh = 1.5 * arg max of a histogram of |x|.

    Parameters
    ----------
    surf : a dictionary with key 'coord',
        surf['coord'] : 2D numpy array of shape (3,v),
            array of coordinates, v is the number of vertices.

    Returns
    -------
    mask : 2D numpy array of shape (1,v), type int, ones and zeros.
    """

    f = (abs(surf['coord'][1,:]) < 50) & \
        (abs(surf['coord'][2,:]-10) < 30) & \
        (abs(surf['coord'][0,:]) < 3)

    # below is equivalent of matlab code:
    # b=(0:0.02:3);
    # h=hist(abs(surf.coord(1,f)),b);

    dx = 0.02
    b = np.arange(0-dx, 3+dx+dx, dx)
    b = b[:-1] + ((b[1:] - b[:-1])/2)
    h = np.histogram(abs(surf['coord'][0, f]), b)[0]

    t = (b[np.where(h==max(h))] +  dx/2) * 1.5

    mask = ~((abs(surf['coord'][0, :]) < t) & f)
    mask = mask.astype('int')

    return mask
