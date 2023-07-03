import numpy as np
from scipy.sparse import csc_matrix
from .edges import edges

def adjacency_matrix(F):
    """
    TO DO
    """
    E = edges(F)
    E = np.sort(E,axis=1)
    E = np.unique(E,axis=0)
    n = np.max(E)+1
    # sparse matrix
    A = csc_matrix((np.ones(E.shape[0]),(E[:,0],E[:,1])),shape=(n,n))
    A = A + A.T
    # if add_diagonal
    #     A = A + csc_matrix((np.ones(n),np.arange(n),np.arange(n)),shape=(n,n))
    return A
