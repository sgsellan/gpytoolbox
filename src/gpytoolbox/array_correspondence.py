import numpy as np

def array_correspondence(A,B,axis=None):
    """Computes a map from A to the equal elements in B

    Parameters
    ----------
    A : (a,) or (a,k) numpy array (must be 1-dim or 2-dim)
    B : (b,) or (b,k) numpy array (must be 1-dim or 2-dim)
    axis : int or None, optional (default None)
        If None, will treat A and B as flat arrays.
        If a number, will check for equality of the entire axis, in which case
        the dimension of A and B across that axis must be equal.

    Returns
    -------
    f : (a,) numpy int array
        index list mapping from A to B, with -1 if there is no
        matching entry.
        If b contains multiple eligible entries, return an arbitrary one.
        If there are no -1s, `b[f] == a`

    Examples
    --------
    TODO
    
    """

    if axis is None:
        A = A.ravel()
        B = B.ravel()

        # Slow for loop
        # f = np.full(A.size, -1, dtype=np.int64)
        # for a in range(A.size):
        #     for b in range(B.size):
        #         if A[a]==B[b]:
        #             f[a] = b

        # While we have to keep track of duplicates in A (to map them to the
        # correct place), we do not care about duplicates in B
        uB,mapB = np.unique(B, return_index=True)
        _,idx,inv = np.unique(np.concatenate((uB,A)),
            return_index=True, return_inverse=True)
        imap = idx[inv[uB.size:]]
        imap[imap>=uB.size] = -1
        f = np.where(imap<0, -1, mapB[imap])

    else:
        assert len(A.shape) == 2
        assert len(B.shape) == 2
        assert axis==-2 or axis==-1 or axis==0 or axis==1
        assert A.shape[axis] == B.shape[axis]

        # This function compares rows, so we reduce the problem to rows here.
        if axis==-2 or axis==0:
                A = A.transpose()
                B = B.transpose()

        # # https://stackoverflow.com/a/64930992 is too memory-intensive
        # inds = (A[None,:,:] == B[:,None,:])
        # i,j = np.nonzero(inds.sum(axis=2) == A.shape[1])
        # f = np.full(A.shape[0], -1, dtype=np.int64)
        # f[j] = i

        # # Slower (but less memory-intensive) with for loops
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
        f = array_correspondence(bytesA,bytesB,axis=None)

    return f


