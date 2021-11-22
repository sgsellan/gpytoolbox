import numpy as np
from numpy.core.function_base import linspace
from scipy.sparse import csr_matrix


def fd_interpolate(P,gs=10,h=(1/9.0),corner=np.array([0.0,0.0])):
    # Given a regular finite-difference grid described by the number of nodes on each side, 
    # the grid spacing, and the location of the bottom-left-front-most corner node, 
    # and a list of points, construct a sparse matrix of bilinear interpolation weights so that P = W @ x
    #
    # Note: This only works in 2D for now
    #
    # Input:
    #       P #P by dim numpy array with interpolated point coordinates
    #       Optional:
    #               gs int grid size
    #               h float spacing between nearest grid nodes
    #               corner a #dim numpy-array of the lowest-valued corner of the grid     
    #
    # Output:
    #       W scipy csr sparse matrix such that if x are the grid nodes, P = W @ x
    # n = floor((P - corner)/h)
    indeces = np.floor( (P - np.tile(corner,(P.shape[0],1)))/h ).astype(int)
    indeces_vectorized = indeces[:,0] + gs*indeces[:,1]
    I = linspace(0,P.shape[0]-1,P.shape[0],dtype=int)
    #I = np.kron(I,np.array([1,1,1,1]))
    I = np.concatenate((I,I,I,I))
    J = np.concatenate(( indeces_vectorized,indeces_vectorized+gs,indeces_vectorized+1,indeces_vectorized+1+gs ))
    # Position in the bottom left corner
    
    vij = np.tile(corner,(P.shape[0],1)) + indeces*h
    vij = (P - vij)/h
    coeff_00 = (1-vij[:,1])*(1-vij[:,0])
    coeff_10 = (1-vij[:,1])*vij[:,0]
    coeff_01 = vij[:,1]*(1-vij[:,0])
    coeff_11 = vij[:,1]*vij[:,0]
    vals = np.concatenate((coeff_00,coeff_01,coeff_10,coeff_11))
    W = csr_matrix((vals,(I,J)),shape=(P.shape[0],gs*gs))
    return W
