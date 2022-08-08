import numpy as np

def normalize_points(v,center=None):
    """Make points fit into a side-length one cube
    
    Translate and scale and arbitrary point set so that it's contained tightly into a 1 by 1 # (by 1) cube, centered at zero by default. Simple yet useful to test code without worrying about scale-dependencies.

    Parameters
    ----------
    v : (n,d) numpy double array
        Matrix of point position coordinates
    center : numpy double array (optional, None by default)
        Where to center the mesh (if None, centered at zero)

    Returns
    -------
    u : numpy double array
        Normalized point position coordinates

    Examples
    --------
    TO-DO
    """
    
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