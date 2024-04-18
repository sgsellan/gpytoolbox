import numpy as np

def tetrahedralize(V,F=None,
    H=None,
    max_volume=None,
    min_rad_edge_ratio=None):
    """Create a tetrahedralization of the input mesh in 3D.

    This is a wrapper for libigl's Tetgen implementation.

    Parameters
    ----------
    V : (n,3) numpy array
        vertex list of points to mesh
    F : (m,3) numpy int array, optional (default None)
        face index list of a triangle.
        If this is None, will simply mesh the convex hull of V.
    H : (h,3) numpy array
        list of seed points inside holes
    max_volume: float, optional (default None)
        If this is not None, the method will refine until this is the maximal
        volume of any tet.
    min_rad_edge_ratio : float, optional (default None)
        If this is not None, the method will refine until this is the minimal
        radius-edge ratio (in radians) of any tet. 
        If this value is too aggressive, the method might silently fail to respect
        the threshold, or hang forever.

    Returns
    -------
    W : numpy double array
        Matrix of mesh vertices
    T : numpy int array
        Matrix of tet indices
    TF : numpy int array
        Matrix of tet face indices

    Notes
    -----
    In the future, we will hopefully use
    "CDT - Constrained Delaunay Tetrahedrization made robust and practical" by
    Diazzi et al. 2023.


    Examples
    --------
    ```python
    import gpytoolbox as gpy
    # Mesh in V,F
    W,T,TF = gpy.copyleft.tetrahedralize(V,F)
    # Tet mesh in W,T,TF
    ```
    """

    # Try to import C++ binding
    try:
        from gpytoolbox_bindings_copyleft import _tetrahedralize_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ tetrahedralize binding.")

    assert len(V.shape)==2 and V.shape[1]==3, "V must be a list of points in 3D."
    assert max_volume is None or max_volume>0, "max_volume must be either None or a positive number."
    if max_volume is None:
        max_volume = -1.
    assert min_rad_edge_ratio is None or min_rad_edge_ratio>=0, "max_angle must be either None or a nonnegative number."
    if min_rad_edge_ratio is None:
        min_rad_edge_ratio = -1.

    if F is None:
        F = np.array([], dtype=int)
    if H is None:
        H = np.array([], dtype=int)

    status,W,T,TF = _tetrahedralize_cpp_impl(V.astype(np.float64),F.astype(np.int32),
        H.astype(np.float64),
        max_volume, min_rad_edge_ratio)
    if status != 0:
        raise RuntimeError("Tetgen failed.")

    return W,T,TF
