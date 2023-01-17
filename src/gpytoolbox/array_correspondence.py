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
    A = A.ravel()
    B = B.ravel()

    # While we have to keep track of duplicates in A (to map them to the
    # correct place), we do not care about duplicates in B
    uB,mapB = np.unique(B, return_index=True, axis=axis)
    _,idx,inv = np.unique(np.concatenate((uB,A)),
        return_index=True, return_inverse=True, axis=axis)
    imap = idx[inv[uB.size:]]
    imap[imap>=uB.size] = -1
    f = np.where(imap<0, -1, mapB[imap])
    
    return f


