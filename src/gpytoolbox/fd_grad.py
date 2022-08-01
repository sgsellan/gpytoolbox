import numpy as np
from scipy.sparse import vstack
from .fd_partial_derivative import fd_partial_derivative

def fd_grad(gs,h):
    """Finite difference gradient on a grid

    Given a regular finite-difference grid described by the number of nodes on each side, the grid spacing and a desired direction, construct a sparse matrix to compute gradients with each component defined on its respective staggered grid

    Parameters
    ----------
    gs : numpy int array
        Grid size [nx,ny(,nz)]
    h : numpy double array
        Spacing between grid points [hx,hy(,hz)]

    Returns
    -------
    G : scipy sparse.csr_matrix
        Sparse matrix of concatenated partial derivatives

    See Also
    --------
    fd_partial_derivative, fd_interpolate.

    Notes
    -----
    For any function f defined on a gs by gs grid, G @ f contains the directional derivatives on a staggered grid, the first gs[1](gs[0]-1) rows are d/dx, while the latter gs[0](gs[1]-1) are d/dy

    Examples
    --------
    TO-DO
    """

    dim = gs.shape[0]
    Dx =  fd_partial_derivative(gs=gs,h=h,direction=0)
    Dy =  fd_partial_derivative(gs=gs,h=h,direction=1)
    if dim==3:
        Dz =  fd_partial_derivative(gs=gs,h=h,direction=2)
        return vstack((Dx,Dy,Dz))
    return vstack((Dx,Dy))