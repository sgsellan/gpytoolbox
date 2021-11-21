import numpy as np

def edge_indeces(n,closed=False):
    if closed:
        return np.reshape(np.concatenate((np.linspace(0,n-1,n,dtype=int),
             np.linspace(1,n-1,n-1,dtype=int),np.array([0]))),(-1, 2),order='F')
    else:
        return np.reshape(np.concatenate((np.linspace(0,n-2,n-1,dtype=int),
             np.linspace(1,n-1,n-1,dtype=int))),(-1, 2),order='F')