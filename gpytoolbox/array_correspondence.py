import numpy as np

def array_correspondence(A,B):
    # Computes a map from the a to the equal elements in b
    #
    # Inputs:
    #       A:  #a numpy array (must be 1-dim)
    #       B:  #b numpy array (must be 1-dim)
    # Outputs:
    #       f  #A numpy index list mapping from the a to b, with -1 if there is
    #          no matching entry.
    #          If b contains multiple eligible entries, return an arbitrary one.
    #          If there are no -1s, b[f] == a

    assert len(A.shape)==1
    assert len(B.shape)==1

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

    return f


