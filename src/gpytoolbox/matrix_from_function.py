# Here I import only the functions I need for these functions
import numpy as np
from scipy.sparse import csc_matrix

def matrix_from_function(fun,X1,X2,sparse=True,sparsity_pattern=None):
    """Returns the two-form matrix given a function and two sets of vectors.

    For a function fun and two sets of vectors X1 and X2, this returns the matrix M whose entries are given by M_{ij} = fun(X1_i,X2_j). This is a useful wrapper for metrics, bilinear forms, distances, kernels, etc.

    Parameters
    ----------
    fun: func
        A vectorized function that accepts two (n,dim) arrays as inputs and returns an (n,) vector
    X1 : (n1,dim) numpy array
        Matrix of first set vector coordinates
    X2 : (n2,dim) numpy array
        Matrix of second set vector coordinates
    sparse : bool, optional (default True)
        Whether to return a sparse matrix (after purging all zero entries). Useful if fun is compactly supported. By default, yes
    sparsity_pattern: [(s,),(s,)] tuple of integer lists, optional (default None)
        If this is not None, then the matrix is assumed to be sparse and the sparsity pattern is given by the two lists. The function will be evaluated only at the elements of the sparsity pattern. This is useful if the function is compactly supported and the sparsity pattern is known. By default, None
    
    Returns
    -------
    M : (n1,n2) numpy array or scipy csc_matrix
        Matrix whose i,j-th entry is fun(X1[i,:],X2[j,:])

    Examples
    --------
    ```python
    def sample_fun(X1,X2):
        return np.linalg.norm(X1-X2,axis=1)
    P1 = np.random.rand(50,1)
    P2 = np.random.rand(71,1)
    M = utility.matrix_from_function(sample_fun,P1,P2)
    ```
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    if sparsity_pattern is not None:
        assert(sparse)
        all_X1 = X1[sparsity_pattern[0],:]
        all_X2 = X2[sparsity_pattern[1],:]
    else:
        all_X1 = np.kron(np.ones((n2,1)),X1)
        all_X2 = np.kron(X2,np.ones((n1,1)))
    # Call to the function
    vals = fun(all_X1,all_X2)

    if sparse:
        if sparsity_pattern is not None:
            I = sparsity_pattern[0]
            J = sparsity_pattern[1]
        else:
            J = np.kron(np.linspace(0,n2-1,n2,dtype=int),np.ones(n1,dtype=int))
            I = np.kron(np.ones(n2,dtype=int),np.linspace(0,n1-1,n1,dtype=int))
        K2 = csc_matrix((vals,(I,J)),shape=(n1,n2))
        K2.eliminate_zeros()
    else:
        K2 = np.reshape(vals,(n1,n2),order='F')
    return K2




