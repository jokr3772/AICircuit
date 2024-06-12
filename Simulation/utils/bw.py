# Find the bandwidth
#
# Desc:
# This files contains utility functions that find the 3-db bandwidh
# by interpolating existing data points.
#
# Author: Yue (Julien) Niu

import numpy as np
from scipy import interpolate


def bw_by_iterp(mag, freq):
    mag_max = np.max(mag)
    mag_3db = 0.707 * mag_max
    
    x = freq
    y = mag - mag_3db
    f = interpolate.UnivariateSpline(x, y, s=0)
    
    roots = f.roots()
    if len(roots) == 1:
        freq_3db = roots[0]
    else:
        freq_3db = roots[1] - roots[0]
    
    return freq_3db
