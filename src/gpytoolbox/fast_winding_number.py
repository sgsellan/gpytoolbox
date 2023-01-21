import numpy as np

def fast_winding_number(Q,V,F):
    """Compute the winding number of a set of query points with respect to a triangle mesh.
    
    Parameters
    ----------
    Q : (n,3) numpy double array
        Matrix of query points
    V : (m,3) numpy double array
        Matrix of mesh vertices
    F : (p,3) numpy int array
        Matrix of triangle indices
    
    Returns
    -------
    W : (n,) numpy double array
        Vector of winding numbers (0 if outside, 1 if inside)
    
    See Also
    --------
    lazy_cage.
    
    Notes
    -----
    This function is a wrapper around the C++ implementation of the winding number algorithm by Barrill et al. (2018).

    Examples
    --------
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj") # Read a mesh
    v = gpytoolbox.normalize_points(v) # Normalize mesh
    # Generate query points
    P = 2*np.random.rand(num_samples,3)-4
    # Compute winding numbers
    W = gpytoolbox.fast_winding_number(P,v,f)
    # W will be zero for points outside the mesh and one for points inside the mesh
    ```
    """

    # Try to import C++ binding
    try:
        from gpytoolbox_bindings import _fast_winding_number_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ fast winding number binding.")

    S = _fast_winding_number_cpp_impl(V.astype(np.float64),F.astype(np.int32),Q.astype(np.float64))

    return S