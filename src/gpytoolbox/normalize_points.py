import numpy as np

def normalize_points(v,center=None):
    # Translate and scale and arbitrary point set so that it's contained
    # tightly into a 1 by 1 # (by 1) cube, centered at zero by default.
    # Simple yet useful to test code without worrying about scale-dependencies
    #
    # Inputs:
    #       v #v by dim numpy array of point position coordinates
    #       Optional:
    #           center dim numpy array where to center mesh (zero by default)
    #
    # Output:
    #       u #v by dim numpy array of output point position coordinates
    #
    
    # Dimension:
    dim = v.shape[1]
    

    # First, move it to the first quadrant:
    for dd in range(dim):
        v[:,dd] = v[:,dd] - np.amin(v[:,dd])
    # Normalize for max length one    
    v = v/np.max(v)
    # Center at zero
    for dd in range(dim):
        v[:,dd] = v[:,dd] - 0.5*np.amax(v[:,dd])

    if (center is not None):
        v = v + np.tile(center,(v.shape[0],1))

    return v