import numpy as np

def volume(V,T):
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
    a = V[T[:,0],:]
    b = V[T[:,1],:]
    c = V[T[:,2],:]
    d = V[T[:,3],:]

    vols = -np.sum(np.multiply(a-d,np.cross(b-c,c-d,axis=1)),axis=1)/6.

    return vols