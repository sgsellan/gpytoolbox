import numpy as np

def point_cloud_to_mesh(P, N,
    method='PSR',
    psr_screening_weight=10.,
    psr_depth=10,
    psr_outer_boundary_type="Neumann",
    verbose=False):
    """Convert a point cloud with normal information to a polyline or triangle
    mesh.

    Parameters
    ----------
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    method : int, optional (default 'PSR')
        which reconstruction method to use.
        Currently, only 'PSR' is supported, which is the official implementation
        of "Screened Poisson Surface Reconstruction" [Kazhdan and Hoppe 2013].
    psr_screening_weight : float, optional (default 1.)
        PSR screening weight
    psr_depth : int, optional (default 8)
        PSR tree depth
    psr_outer_boundary_type : string, optional (default "Neumann")
        The boundary condition to use for the outer boundary in PSR.
        Valid options are "Neumann" and "Dirichlet".
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running
    
    Returns
    -------
    V : (n,d) numpy array
        reconstructed vertices
    F : (m,d) numpy array
        reconstructed polyline / triangle mesh indices

    See Also
    --------
    poisson_surface_reconstruction.

    Notes
    -----
    Gpytoolbox makes some choices for default parameters which are not the
    same as in the official PSR paper or repository.

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    # Normal data in P,N
    P = np.load("points.npy")
    V,F = gpy.point_cloud_to_mesh(P,N)
    """

    assert P is not None and N is not None, "No point cloud provided."
    assert P.shape==N.shape, "Must supply one normal for every point."
    assert len(P)>1 and len(P.shape)==2 and P.shape[0]>1, "Must supply at least two points."
    dim = P.shape[1]
    assert dim==2 or dim==3, "Only dimensions 2 and 3 supported."

    if method=='PSR':
        # Try to import C++ binding
        try:
            from gpytoolbox_bindings import _point_cloud_to_mesh_cpp_impl
        except:
            raise ImportError("Gpytoolbox cannot import its C++ point_cloud_to_mesh binding.")

        assert psr_depth>0, "Depth must be a positive integer."
        assert psr_screening_weight>=0., "Screening weight must be a nonnegative scalar."

        # Currently, parallelizing this can crash the PSR code, so we do not
        # enable this feature.
        parallel = False

        # Max depths supported by PSR
        psr_depth = min(12, psr_depth)

        # TODO: We have C++ implementations for both double and float, do we want
        # to use this?
        V,F = _point_cloud_to_mesh_cpp_impl(P.astype(np.float64),
            N.astype(np.float64),
            psr_screening_weight,
            psr_depth,
            psr_outer_boundary_type,
            parallel, verbose)
    else:
        assert False, "This point cloud reconstruction method is not supported."

    return V,F


