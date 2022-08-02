import numpy as np

def in_element_aabb(queries,V,F):
    """Finite element gradient matrix

    Given a triangle mesh or a polyline, computes the finite element gradient matrix assuming piecewise linear hat function basis.

    Parameters
    ----------
    queries : numpy double array
        Matrix of query point coordinates
    V : numpy double array
        Matrix of vertex coordinates
    F : numpy int array, optional (default None)
        Matrix of triangle (2D) or tetrahedron (3D) indices

    Returns
    -------
    I : numpy int array
        Vector of indeces into F of the elements that contain each query point (-1 means no containing element).

    See Also
    --------

    Notes
    -----

    Examples
    --------
    TO-DO
    """
    try:
        from gpytoolbox_bindings import _in_element_aabb_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ decimate binding.")

    

    I = _in_element_aabb_cpp_impl(queries,V,F.astype(np.int32))

    return I
