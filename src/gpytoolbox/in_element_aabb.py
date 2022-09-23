import numpy as np

def in_element_aabb(queries,V,F):
    """Find which triangle or tet a set of query points lay on

    Given a set of query points and a triangle or tetrahedral mesh, construct an AABB data structure to learn which specific mesh elements each query point is in.

    Parameters
    ----------
    queries : numpy double array
        Matrix of query point coordinates
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) or (m,4) numpy int array
        face/tet index list of a triangle/tet mesh

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
    ```python
    V,F = gpytoolbox.regular_square_mesh(23)
    num_samples = 100
    queries = np.random.rand(num_samples,3)
    I = in_element_aabb(queries,V,F)
    ```
    """
    try:
        from gpytoolbox_bindings import _in_element_aabb_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    

    I = _in_element_aabb_cpp_impl(queries,V,F.astype(np.int32))

    return I
