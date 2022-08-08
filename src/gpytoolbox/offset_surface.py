import numpy as np

def offset_surface(V,F,iso,grid_size=100):
    """Compute the surface obtained from dilating a given mesh

    Given a triangle mesh, paste signed distances onto a grid, then use Marching Cubes to obtain a new surface at a given isolevel

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    iso : double
        Matrix of vertex coordinates of the second mesh
    grid_size : int, optional (default 100)
        Size of the grid used to compute signed distances

    Returns
    -------
    U : numpy double array
        Matrix of output vertex coordinates
    G : numpy int array
        Matrix of output triangle indices
    
    See Also
    --------
    lazy_cage.

    Examples
    --------
    TO-DO
    """


    try:
        from gpytoolbox_bindings import _offset_surface_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    v, f = _offset_surface_cpp_impl(V,F.astype(np.int32),iso,grid_size)

    return v,f
