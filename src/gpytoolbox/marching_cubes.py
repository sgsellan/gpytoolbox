import numpy as np

def marching_cubes(S,GV,nx,ny,nz,isovalue=0.0):
    """Compute the marching cubes of a scalar field.
    
    Parameters
    ----------
    S : (n,) numpy double array
        Vector of scalar values
    GV : (n,3) numpy double array
        Matrix of grid vertices
    nx : int
        Number of grid vertices in x direction
    ny : int
        Number of grid vertices in y direction
    nz : int
        Number of grid vertices in z direction
    isovalue : double, optional (default: 0)
        Isovalue to use for reconstruction
        
    Returns
    -------
    V : (m,3) numpy double array
        Matrix of mesh vertices
    F : (p,3) numpy int array
        Matrix of triangle indices
    
    See Also
    --------
    lazy_cage, fast_winding_number, squared_distance

    Examples
    --------
    ```python
    # Some scalar function fun
    def fun(V):
        return np.sum(V**2,axis=1)
    # Generate a grid
    GV,_ = gpytoolbox.regular_cube_mesh(100)
    # Evaluate scalar function on grid
    S = fun(GV)
    # Compute isosurface
    V,F = gpytoolbox.marching_cubes(S,GV,100,100,100,0.5)
    ```
    """
    # Try to import C++ binding
    try:
        from gpytoolbox_bindings import _marching_cubes_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ marching cubes binding.")

    V,F = _marching_cubes_cpp_impl(S.astype(np.float64),GV.astype(np.float64),nx,ny,nz,isovalue)

    return V,F
