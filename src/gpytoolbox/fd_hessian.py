import numpy as np
from scipy.sparse import lil_matrix, vstack

def fd_hessian(gs, h):
    """
    Computes the finite difference Hessian matrix for a 2D grid, 
    only at interior nodes (boundary derivatives are zero), using sparse matrices.

    Parameters:
        gs (array-like): Grid size as [nx, ny].
        h (array-like): Grid spacing as [hx, hy].

    Returns:
        H (csr_matrix): A sparse matrix of shape (4 * (nx * ny), nx * ny),
                        representing the Hessian operators concatenated vertically.
    """
    nx, ny = gs
    hx, hy = h
    num_points = nx * ny

    # Initialize sparse matrices (using List of Lists format for fast construction)
    Dxx = lil_matrix((num_points, num_points))
    Dyy = lil_matrix((num_points, num_points))
    Dxy = lil_matrix((num_points, num_points))

    # Flattened grid indices (using np.arange to quickly access index locations)
    indices = np.arange(num_points).reshape(ny, nx)

    # Compute indices for interior grid points
    interior_x = np.arange(1, nx - 1)
    interior_y = np.arange(1, ny - 1)

    # Vectorized: compute the second derivative and cross derivative terms
    for j in interior_y:
        for i in interior_x:
            idx = indices[j, i]

            # Dxx (second derivative in x)
            Dxx[idx, indices[j, i]] = -2 / (hx**2)
            Dxx[idx, indices[j, i - 1]] = 1 / (hx**2)
            Dxx[idx, indices[j, i + 1]] = 1 / (hx**2)

            # Dyy (second derivative in y)
            Dyy[idx, indices[j, i]] = -2 / (hy**2)
            Dyy[idx, indices[j - 1, i]] = 1 / (hy**2)
            Dyy[idx, indices[j + 1, i]] = 1 / (hy**2)

            # Dxy (cross derivative in x and y)
            Dxy[idx, indices[j + 1, i + 1]] = 1 / (4 * hx * hy)
            Dxy[idx, indices[j - 1, i - 1]] = 1 / (4 * hx * hy)
            Dxy[idx, indices[j + 1, i - 1]] = -1 / (4 * hx * hy)
            Dxy[idx, indices[j - 1, i + 1]] = -1 / (4 * hx * hy)

    # Dyx is the same as Dxy for symmetric Hessians
    Dyx = Dxy.copy()

    # Convert the sparse matrices to CSR format before stacking
    Dxx_csr = Dxx.tocsr()
    Dyy_csr = Dyy.tocsr()
    Dxy_csr = Dxy.tocsr()
    Dyx_csr = Dyx.tocsr()

    # Stack the matrices vertically in sparse format
    H = vstack([Dxx_csr, Dxy_csr, Dyx_csr, Dyy_csr])

    return H