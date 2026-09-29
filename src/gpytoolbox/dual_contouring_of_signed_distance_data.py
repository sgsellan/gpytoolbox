import numpy as np

from gpytoolbox_bindings import _dcsdd_cpp_impl


def dual_contouring_of_signed_distance_data(
    S,
    GV,
    nx,
    ny,
    nz,
    isovalue=0.0,
    outer_iters=100,
    inner_iters=100,
    batch_size=200000,
    hermite_update=True,
    mu=0.1,
    dc_weight=0.02,
    svd_threshold=0.01,
    new_hermite_pos_weight=0.2,
    new_face_pos_weight=0.2,
    new_hermite_normal_weight=0.2,
    verbose=False,
):
    """Reconstructs a quad mesh from regularly sampled 3D signed distance data,
    using the method described in 'Dual Contouring of Signed Distance Data', by
    X. Carrera, N. Wang, C. Batty, O. Stein and S. Sellán [2026].

    Parameters
    ----------
    S : array_like, shape (n,)
        sdf(GV), i.e., signed distance values at the grid vertices.
    GV : array_like, shape (n, 3)
        Grid vertex positions. Row i + nx*(j + ny*k) must be grid vertex
        (i,j,k), i.e., the first axis must vary fastest, then the second,
        then the third (see the examples below).
    nx : int
        Number of grid vertices along the first axis.
    ny : int
        Number of grid vertices along the second axis.
    nz : int
        Number of grid vertices along the third axis.
    isovalue : float, optional (default 0.0)
        Level set to extract.
    outer_iters : int, optional (default 100)
        Number of iterations in the outer loop.
    inner_iters : int, optional (default 100)
        Number of iterations in the inner loop (local energy minimization).
    batch_size : int, optional (default 200000)
        Maximum number of SDF grid points processed per batch.
    hermite_update : bool, optional (default True)
        Whether to refine Hermite positions from the mesh.
    mu : float, optional (default 0.1)
        Regularization weight.
    dc_weight : float, optional (default 0.02)
        Weight of the Dual Contouring (Hermite) energy term.
    svd_threshold : float, optional (default 0.01)
        Singular values below this are dropped when solving each cell's QEF.
    new_hermite_pos_weight : float, optional (default 0.2)
        Blend weight for updating Hermite positions.
    new_face_pos_weight : float, optional (default 0.2)
        Blend weight for updating face positions.
    new_hermite_normal_weight : float, optional (default 0.2)
        Blend weight for updating Hermite normals.
    verbose : bool, optional (default False)
        Whether to print progress information.

    Returns
    -------
    V : numpy.ndarray, shape (m, 3)
        Reconstructed vertex positions.
    F : numpy.ndarray, shape (p, 4)
        Reconstructed quadrilateral faces.

    Examples
    --------
    ```python
    nx, ny, nz = 32, 32, 32
    x = np.linspace(-1., 1., nx)
    y = np.linspace(-1., 1., ny)
    z = np.linspace(-1., 1., nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    GV = np.stack((X.ravel(order='F'), Y.ravel(order='F'),
        Z.ravel(order='F')), axis=-1)
    S = fun(GV) # Some SDF fun
    V, F = gpytoolbox.dual_contouring_of_signed_distance_data(S, GV, nx, ny, nz)
    ```

    ```python
    nx, ny, nz = 32, 32, 32
    GV, _ = gpytoolbox.regular_cube_mesh(nx, ny, nz)
    GV = np.stack([
        GV[:, d].reshape((ny, nx, nz)).transpose(1, 0, 2).ravel(order='F')
        for d in range(3)
    ], axis=-1)
    S = fun(GV) # Some SDF fun
    V, F = gpytoolbox.dual_contouring_of_signed_distance_data(S, GV, nx, ny, nz)
    ```
    """
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    GV = np.asarray(GV, dtype=np.float64)

    nx = int(nx)
    ny = int(ny)
    nz = int(nz)

    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("nx, ny, and nz must all be at least 2.")

    if GV.ndim != 2 or GV.shape[1] != 3:
        raise ValueError("GV must have shape (n, 3).")

    if S.shape[0] != GV.shape[0]:
        raise ValueError(
            "GV and S must contain the same number of samples."
        )

    expected_samples = nx * ny * nz
    if S.shape[0] != expected_samples:
        raise ValueError(
            "GV and S must contain nx * ny * nz samples; "
            f"expected {expected_samples}, got {S.shape[0]}."
        )

    if outer_iters < 0:
        raise ValueError("outer_iters must be nonnegative.")

    if inner_iters < 0:
        raise ValueError("inner_iters must be nonnegative.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    return _dcsdd_cpp_impl(
        S=S,
        GV=GV,
        nx=nx,
        ny=ny,
        nz=nz,
        isovalue=float(isovalue),
        outer_iters=int(outer_iters),
        inner_iters=int(inner_iters),
        hermite_update=bool(hermite_update),
        mu=float(mu),
        dc_weight=float(dc_weight),
        svd_threshold=float(svd_threshold),
        new_hermite_pos_weight=float(new_hermite_pos_weight),
        new_face_pos_weight=float(new_face_pos_weight),
        new_hermite_normal_weight=float(new_hermite_normal_weight),
        batch_size=int(batch_size),
        verbose=bool(verbose),
    )
