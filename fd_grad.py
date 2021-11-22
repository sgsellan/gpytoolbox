import numpy as np
from scipy.sparse import vstack
from fd_partial_derivative import fd_partial_derivative

def fd_grad(gs=np.array([10,10]),h=np.array([1/9.0,1/9.0])):
    # Given a regular finite-difference grid described by the number of nodes 
    # on each side, the grid spacing and a desired direction, construct a sparse matrix 
    # to compute gradients with each component defined on its respective staggered grid
    # 
    #
    # Note: This works for 2D only
    #
    # Input:
    #       Optional:
    #               gs #dim int numpy array of grid sizes [nx,ny]
    #               h #dim float numpy array of spacing between nearest grid nodes [hx,hy]
    #
    # Output:
    #       W scipy csr sparse matrix such that for any function f defined on a gs by gs grid, 
    #           then G @ f contains the directional derivatives on a staggered grid, the first
    #           gs(gs-1) rows are d/dx, while the latter gs(gs-1) are d/dy
    # 
    # 
    Dx =  fd_partial_derivative(gs=gs,h=h,direction=0)
    Dy =  fd_partial_derivative(gs=gs,h=h,direction=1)
    return vstack((Dx,Dy))