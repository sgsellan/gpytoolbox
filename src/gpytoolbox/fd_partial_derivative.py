import numpy as np
from numpy.core.function_base import linspace
from scipy.sparse import csr_matrix


def fd_partial_derivative(gs,h,direction):
    """Finite difference partial derivative on a grid

    Given a regular finite-difference grid described by the number of nodes on each side, the grid spacing and a desired direction, construct a sparse matrix to compute first partial derivatives in the given direction onto the staggered grid in that direction.

    Parameters
    ----------
    gs : numpy int array
        Grid size [nx,ny(,nz)]
    h : numpy double array
        Spacing between grid points [hx,hy(,hz)]
    direction : int
        Direction with respect to which the derivative is computed (x: 0, y: 1, z: 2)

    Returns
    -------
    W : scipy sparse.csr_matrix
        Sparse matrix of partial derivative

    See Also
    --------
    fd_grad, fd_interpolate.

    Notes
    -----
    For any function f defined on a gs by gs grid, then W @ f contains the directional derivative on a staggered grid

    Examples
    --------
    TO-DO
    """
    # Given a regular finite-difference grid described by the number of nodes 
    # on each side, the grid spacing and a desired direction, construct a sparse matrix 
    # to compute first partial derivatives in the given direction onto the 
    # staggered grid in that direction.
    #
    # Note: This works for 2D only
    #
    # Input:
    #       Optional:
    #               gs #dim int numpy array of grid sizes [nx,ny]
    #               h #dim float numpy array of spacing between nearest grid nodes [hx,hy]
    #               direction integer index of direction (x is 0, y is 1)
    #
    # Output:
    #       W scipy csr sparse matrix such that for any function f defined on a gs by gs grid, 
    #           then W @ f contains the directional derivative on a staggered grid
    # 
    #  

    dim = gs.shape[0]
    new_grid = gs.copy()
   
    if direction==0:
        new_grid[0] = new_grid[0]-1
        next_ind = 1
    elif direction==1:
        new_grid[1] = new_grid[1]-1
        next_ind = gs[0]
    elif direction==2:
        new_grid[2] = new_grid[2]-1
        next_ind = gs[0]*gs[1]
    
    num_vertices = np.prod(new_grid)

    # dimension-dependent part
    if dim==2:
        xi, yi = np.meshgrid(linspace(0,new_grid[0]-1,new_grid[0],dtype=int),linspace(0,new_grid[1]-1,new_grid[1],dtype=int))
        vectorized_indeces = np.reshape(xi,(-1, 1)) + gs[0]*np.reshape(yi,(-1, 1))
    elif dim==3:
        xi, yi, zi = np.meshgrid(linspace(0,new_grid[0]-1,new_grid[0],dtype=int),linspace(0,new_grid[1]-1,new_grid[1],dtype=int),linspace(0,new_grid[2]-1,new_grid[2],dtype=int),indexing='ij')
        vectorized_indeces = np.reshape(xi,(-1, 1),order='F') + gs[0]*(np.reshape(yi,(-1, 1),order='F') +gs[1]* np.reshape(zi,(-1, 1),order='F'))

    J = np.squeeze(np.concatenate((vectorized_indeces,vectorized_indeces+next_ind)))
    I = np.concatenate((linspace(0,num_vertices - 1,num_vertices,dtype=int),linspace(0,num_vertices - 1,num_vertices,dtype=int)))


    vals = np.concatenate(( -np.ones((num_vertices)), np.ones((num_vertices))))

    D = csr_matrix((vals,(I,J)),shape=(num_vertices,np.prod(gs)))/h[direction]


    return D