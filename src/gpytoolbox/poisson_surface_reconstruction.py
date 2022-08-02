import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.ndimage.filters import gaussian_filter
from .fd_grad import fd_grad
from .fd_interpolate import fd_interpolate



def poisson_surface_reconstruction(P,N,gs=None,h=None,corner=None,screened=False):
    """Reconstruct surface from point cloud
    
    Given an oriented point cloud on a volume's surface, return the values on a regular grid of an implicit function that represents the volume enclosed by the surface.

    Parameters
    ----------
    P : numpy double array
        Matrix of point cloud coordinates
    N : numpy double array 
        Matrix of unit-norm normals
    gs : numpy int array, optional (default None)
        Vector of grid sizes [nx,ny]
    h : numpy double array, optional (default None)
        Vector of spacing between nearest grid 
    corner : numpy double array, optional (default None)
        Vector of lowest-valued corner of the grid     
    screened : bool, optional (default False)
        Whether to used *screened* Poisson Surface Reconstruction

    Returns
    -------
    g : numpy double array
        Vector of implicit function values on the requested grid, ordered increasing and row first (see poisson_surface_reconstruction_unit_test.py) 
    sigma : double
        The isolvalue of the surface; i.e. to be deducted from g if one wants the zero-level-set to cross the surface.

    Notes
    -----
    This only works in 2D. This only outputs the grid implicit values, not a mesh

    Examples
    --------
    TODO
    """

    # Gradient matrix
    G = fd_grad(gs=gs,h=h)
    # Interpolator in main grid
    W = fd_interpolate(P,gs=gs,h=h,corner=corner)
    # First: estimate sampling density and weigh normals by it:
    image = W.T @ np.ones((N.shape[0],1))
    image_blurred = gaussian_filter(np.reshape(image,(gs[0],gs[1]),order='F'),sigma=7)
    image_blurred_vectorized = np.reshape(image_blurred,(gs[0]*gs[1],1),order='F')
    weights = W @ image_blurred_vectorized
    N = N/np.hstack((weights,weights))
    # Build staggered grids
    corner_x = corner + np.array([0.5*h[0],0.0])
    corner_y = corner + np.array([0.0,0.5*h[1]])
    gs_x = gs-np.array([1,0],dtype=int)
    gs_y = gs-np.array([0,1],dtype=int)

    # Interpolators on staggered grids
    Wx = fd_interpolate(P,gs=gs_x,h=h,corner=corner_x)
    Wy = fd_interpolate(P,gs=gs_y,h=h,corner=corner_y)
    Nx = Wx.T @ N[:,0]
    Ny = Wy.T @ N[:,1]
    v = np.concatenate((Nx,Ny))
    rhs = G.T @ v # right hand side in linear equation
    if screened:
        lhs = G.T @ G  +  4.0*W.T @ W # left hand side in linear equation
    else:
        lhs = G.T @ G # left hand side in linear equation
    g = spsolve(lhs,rhs)
    # Good isovalue
    weights = np.squeeze(weights)
    sigma = np.sum((W @ g)*(1/weights) ) / np.sum(1/weights)
    return g, sigma