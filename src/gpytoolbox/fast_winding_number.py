import numpy as np

def fast_winding_number(Q,V,F,fwn_bvh=None):
    """Compute the winding number of a set of query points with respect to a triangle mesh.

    Parameters
    ----------
    Q : (n,3) numpy double array
        Matrix of query points
    V : (m,3) numpy double array
        Matrix of mesh vertices
    F : (p,3) numpy int array
        Matrix of triangle indices
    fwn_bvh : gpytoolbox.FastWindingNumberBVH, optional (default None)
        Precomputed BVH built via `gpytoolbox.FastWindingNumberBVH(V, F)`.
        If provided, reuses the BVH across calls instead of rebuilding it,
        avoiding the O(n) precomputation cost on every query.

    Returns
    -------
    W : (n,) numpy double array
        Vector of winding numbers (0 if outside, 1 if inside)

    See Also
    --------
    lazy_cage, FastWindingNumberBVH.

    Notes
    -----
    This function is a wrapper around the C++ implementation of the winding number algorithm by Barrill et al. (2018).

    Examples
    --------
    Standard one-shot call (rebuilds the BVH internally each time):
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj") # Read a mesh
    v = gpytoolbox.normalize_points(v) # Normalize mesh
    # Generate query points
    P = 2*np.random.rand(num_samples,3)-4
    # Compute winding numbers
    W = gpytoolbox.fast_winding_number(P,v,f)
    # W will be zero for points outside the mesh and one for points inside the mesh
    ```

    When making many calls against the same mesh, build the BVH once and
    reuse it via the `fwn_bvh=` kwarg to avoid the O(n) precomputation cost
    on every call:
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj")
    v = gpytoolbox.normalize_points(v)
    bvh = gpytoolbox.FastWindingNumberBVH(v, f) # build once
    for _ in range(num_iters):
        P = 2*np.random.rand(num_samples,3)-4
        W = gpytoolbox.fast_winding_number(P,v,f,fwn_bvh=bvh)
    ```
    """

    if fwn_bvh is not None:
        return fwn_bvh.winding_number(Q.astype(np.float64))

    # Try to import C++ binding
    try:
        from gpytoolbox_bindings import _fast_winding_number_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ fast winding number binding.")

    S = _fast_winding_number_cpp_impl(V.astype(np.float64),F.astype(np.int32),Q.astype(np.float64))

    return S
