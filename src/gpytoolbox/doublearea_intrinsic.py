import numpy as np


def doublearea_intrinsic(l_sq,F):
    """Construct the doublearea of each element of a line or triangle mesh.

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    dblA : (m,) numpy double array
        vector of twice the (unsigned) area/length 

    See Also
    --------
    doublearea.

    Examples
    --------
     ```python
    # Mesh in V,F
    from gpytoolbox import halfedge_lengths_squared, doublearea_intrinsic
    l = halfedge_lengths_squared(V,F)
    L = doublearea_intrinsic(l_sq,F)
    ```
    """

    assert F.shape == l_sq.shape
    assert F.shape[1]==3
    assert np.all(l_sq >= 0)

    l = np.sqrt(l_sq)

    # Using Kahan's formula
    # https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
    a,b,c = l[:,0], l[:,1], l[:,2]
    # previously (gave NaNs for very small triangles)
    # dblA = 0.5 * np.sqrt((a+(b+c)) * (c-(a-b)) * (c+(a-b)) * (a+(b-c)))
    arg = (a+(b+c)) * (c-(a-b)) * (c+(a-b)) * (a+(b-c))
    dblA = 0.5 * np.sqrt(np.maximum(arg, 0.))
    
    return dblA
