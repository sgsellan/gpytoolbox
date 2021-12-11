import numpy as np

def edge_indeces(n,closed=False):
    # Given an ordered polyline, this returns the edge indeces in a similar way
    # to how the face indeces of a triangle mesh are given.
    # Inputs:
    #       n integer number of vertices
    #       Optional:
    #           closed boolean with whether to close the path or not
    # Outputs:
    #       EC #n-1 (#n if closed) by 2 matrix of edge indeces

    # Note: the "order='F'" in all the reshapes is to mimic scanning order from
    # Matlab, otherwise it does the reverse order (rows first)
    if closed:
        return np.reshape(np.concatenate((np.linspace(0,n-1,n,dtype=int),
             np.linspace(1,n-1,n-1,dtype=int),np.array([0]))),(-1, 2),order='F')
    else:
        return np.reshape(np.concatenate((np.linspace(0,n-2,n-1,dtype=int),
             np.linspace(1,n-1,n-1,dtype=int))),(-1, 2),order='F')