import numpy as np
from numpy.core.function_base import linspace
from scipy.sparse.linalg import spsolve
from fd_grad import fd_grad
from fd_interpolate import fd_interpolate

def poisson_surface_reconstruction(P,N,gs=np.array([10,10]),h=np.array([1/9.0,1/9.0]),corner=np.array([0.0,0.0])):
    # Given an oriented pointcloud on a volume's surface, return the values on a regular grid of an implicit function
    # that represents the volume enclosed by the surface.
    #
    # Note: This only works in 2D
    # Note: This only outputs the grid implicit values, not a mesh
    #
    # Input:
    #       P #P by dim numpy array of point coordinates
    #       N #P by dim numpy array of unit-norm normals  
    #       Optional:
    #               gs #dim int numpy array of grid sizes [nx,ny]
    #               h #dim float numpy array of spacing between nearest grid nodes [hx,hy]
    #               corner a #dim numpy-array of the lowest-valued corner of the grid     
    #
    # Output:
    #       g np array vector of implicit function values on the requested grid, ordered 
    #           increasing and row first (see poisson_surface_reconstruction_unit_test.py)
    #       sigma is the isolvalue of the surface; i.e. to be deducted from g if one wants
    #           the zero-level-set to cross the surface

    # Gradient matrix
    G = fd_grad(gs=gs,h=h)
    # Interpolator in main grid
    W = fd_interpolate(P,gs=gs,h=h,corner=corner)
    
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
    distributed_normals = np.hstack([Nx,Ny])
    #print(distributed_normals)
    v = np.concatenate((Nx,Ny))
    rhs = G.T @ v # right hand side in linear equation
    lhs = G.T @ G # left hand side in linear equation
    g = spsolve(lhs,rhs)
    # Good isovalue
    sigma = np.sum(W @ g) / P.shape[0]
    return g, sigma