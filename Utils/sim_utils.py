# Analyze results
#
# Desc:
# Analyze results such as training loss, training and testing error
#
# Author: Yue(Julien) Niu

import numpy as np
from scipy import interpolate


def calc_hist(results):
    """calculate histograms of results
    :param results: simulation result data
    """
    data = {}
    for key in results[0]:
        if 'Error' in key:
            data[key] = []
            
    for item in results:
        for key in item:
            if 'Error' in key:
                data[key].append(item[key])
                
    # convert to numpy array and calculate histograms
    for key in data:
        data[key] = np.array(data[key])
        cnt, bin = np.histogram(data[key], bins=30, density=True)
        
        print('\n{}:\n'.format(key))
        for bin_i, cnt_i in zip(bin, cnt):
            print('{:.3f}, {:.3f}'.format(bin_i, cnt_i))


# Find the bandwidth
#
# Desc:
# utility function that finds the 3-db bandwidh
# by interpolating existing data points.
#
# Author: Yue (Julien) Niu

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