import numpy as np
import scipy as sp
from gpytoolbox.array_correspondence import array_correspondence

def row_correspondence(A,B):
    # Computes a map from the rows of A to the rows of B that are equal
    #
    # TODO: remove sparse dependency and vectorize over columns.
    #
    # Inputs:
    #       A:  #A by k numpy matrix
    #       B:  #B by k numpy matrix.
    # Outputs:
    #       f  #A numpy index list mapping from the rows of A to the rows of B
    #          that are equal, with -1 if there is no matching row.
    #          If B contains multiple eligible rows, return an arbitrary one.
    #          If there are no -1s, B[f,:] == A

    assert A.shape[1] == B.shape[1]

    # # https://stackoverflow.com/a/64930992 is too memory-intensive
    # inds = (A[None,:,:] == B[:,None,:])
    # i,j = np.nonzero(inds.sum(axis=2) == A.shape[1])
    # f = np.full(A.shape[0], -1, dtype=np.int64)
    # f[j] = i

    # # Slower (but less memory-intensive) with for loops
    # i,j = [],[]
    # for a in range(A.shape[0]):
    #     for b in range(B.shape[0]):
    #         if (A[a,:]==B[b,:]).all():
    #             i.append(b)
    #             j.append(a)
    # f = np.full(A.shape[0], -1, dtype=np.int64)
    # f[j] = i

    # # Shorter, slower (but less memory-intensive) with for loops
    # f = np.full(A.shape[0], -1, dtype=np.int64)
    # for a in range(A.shape[0]):
    #     for b in range(B.shape[0]):
    #         if (A[a,:]==B[b,:]).all():
    #             f[a] = b

    # Convert each row to bytes, intersect byte arrays in 1d
    def to_byte_array(x):
        # Adapted from https://stackoverflow.com/a/54683422
        dt = np.dtype('S{:d}'.format(x.shape[1] * x.dtype.itemsize))
        return np.frombuffer(x.tobytes(), dtype=dt)
    bytesA = to_byte_array(A)
    bytesB = to_byte_array(B.astype(A.dtype))
    f = array_correspondence(bytesA,bytesB)

    return f


