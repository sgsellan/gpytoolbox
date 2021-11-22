import numpy as np
from numpy.core.function_base import linspace
from scipy.sparse import csr_matrix


def fd_partial_derivative(gs=10,h=(1/9.0),direction=0):
    # Given a regular finite-difference grid described by the number of nodes 
    # on each side, the grid spacing and a desired direction, construct a sparse matrix 
    # to compute first partial derivatives in the given direction onto the 
    # staggered grid in that direction.
    #
    # Note: This works for 2D only
    #
    # Input:
    #       Optional:
    #               gs int grid size
    #               h float spacing between nearest grid nodes
    #               direction integer index of direction (x is 0, y is 1)
    #
    # Output:
    #       W scipy csr sparse matrix such that for any function f defined on a gs by gs grid, 
    #           then W @ f contains the directional derivative on a staggered grid
    # 
    #  
    if direction==0:
        # First, get the indeces of all possible bottom-left corners
        # which are 1:(gs-2) + gs*(1:(gs-2))
        # One of these is a kron prod, the other is a repetition
        horizontal_indeces = linspace(0,gs-2,gs-1,dtype=int)
        vertical_indeces = linspace(0,gs-1,gs,dtype=int)
        #print(horizontal_indeces) 
        #print(vertical_indeces) 
        horizontal_indeces_rep = np.tile(horizontal_indeces,gs)
        vertical_indeces_rep = np.kron(vertical_indeces,np.ones((gs-1),dtype=int))
        #print(horizontal_indeces_rep) 
        #print(vertical_indeces_rep) 
        vectorized_indeces = horizontal_indeces_rep + gs*vertical_indeces_rep
        #print(vectorized_indeces)
        J = np.concatenate((vectorized_indeces,vectorized_indeces+1))
        I = np.concatenate((linspace(0,(gs-1)*gs - 1,(gs-1)*gs,dtype=int),linspace(0,(gs-1)*gs - 1,(gs-1)*gs,dtype=int)))
        vals = np.concatenate(( -np.ones((gs*(gs-1))), np.ones((gs*(gs-1)))  ))
        # Build scipy matrix
        D = csr_matrix((vals,(I,J)),shape=(gs*(gs-1),gs*gs))/h
    elif direction==1:
        # First, get the indeces of all possible bottom-left corners
        # which are 1:(gs-2) + gs*(1:(gs-2))
        # One of these is a kron prod, the other is a repetition
        vertical_indeces = linspace(0,gs-2,gs-1,dtype=int)
        horizontal_indeces = linspace(0,gs-1,gs,dtype=int)
        #print(horizontal_indeces) 
        #print(vertical_indeces) 
        horizontal_indeces_rep = np.tile(horizontal_indeces,gs-1)
        vertical_indeces_rep = np.kron(vertical_indeces,np.ones((gs),dtype=int))
        #print(horizontal_indeces_rep) 
        #print(vertical_indeces_rep) 
        vectorized_indeces = horizontal_indeces_rep + gs*vertical_indeces_rep
        #print(vectorized_indeces)
        J = np.concatenate((vectorized_indeces,vectorized_indeces+gs))
        I = np.concatenate((linspace(0,(gs-1)*gs - 1,(gs-1)*gs,dtype=int),linspace(0,(gs-1)*gs - 1,(gs-1)*gs,dtype=int)))
        vals = np.concatenate(( -np.ones((gs*(gs-1))), np.ones((gs*(gs-1)))  ))
        # Build scipy matrix
        D = csr_matrix((vals,(I,J)),shape=(gs*(gs-1),gs*gs))/h   

    return D