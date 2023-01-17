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
    if axis not in (None, 0, 1):
        raise Exception("Axis can only be None, 0, 1")
    if len(A.shape) > 2 or len(B.shape) > 2:
        raise Exception("Inputs A, B can only be up to 2 dimensional")

    if axis == 1:
        # np.unique behaves weird with axis=1... but we only work with
        # up to dim 2, so always simply use the axis=0 case.
        A = A.T
        B = B.T
        axis = 0

    # While we have to keep track of duplicates in A (to map them to the
    # correct place), we do not care about duplicates in B

    # uB is deduplicated B. mapB allows us to map to the first occurence of each vector.
    uB, mapB = np.unique(B, return_index=True, axis=axis)

    # We concatenate uB with A. Any entries from A that get de-duped is a 'hit'.
    _, idx, inv = np.unique(np.concatenate((uB,A), axis=axis),
        return_index=True, return_inverse=True, axis=axis)

    # We don't care about the range of the output that is about uB- that was a decoy to
    # grill out the hits from A.
    imap = idx[inv[uB.shape[0]:]]
    imap[imap>=uB.shape[0]] = -1
    f = np.where(imap<0, -1, mapB[imap])

    return f


