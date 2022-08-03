import numpy as np

def volume(v,t):
    """Computes a vector containing the volumes of all tetrahedra in a mesh

    Parameters
    ----------
    V : numpy double array
        Matrix of mesh vertex position coordinates
    T : numpy double array
        Matrix of mesh tetrahedra indices into V

    Returns
    -------
    vols : numpy double array
        Vector of per-tetrahedron volumes

    See Also
    --------
    doublearea

    Examples
    --------
    TODO
    """

    
    # Dimension:
    a = v[t[:,0],:]
    b = v[t[:,1],:]
    c = v[t[:,2],:]
    d = v[t[:,3],:]

    vols = -np.sum(np.multiply(a-d,np.cross(b-c,c-d,axis=1)),axis=1)/6.

    return vols