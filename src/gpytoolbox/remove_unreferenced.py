import numpy as np

def remove_unreferenced(V, F,
    return_maps=False):
    """Removes vertices from a vertex-list-face-list polyline or mesh that are
    unreferenced in the face list.
    This can be useful to prevent, for example, a mass matrix that is not
    invertible because of vertices that are declared in a vertex list, but never
    used in the face list.

    This function will silently skip `-1` entries in `F`.
    You can use this, for example, to mark invalid vertices that are to be
    skipped.


    Parameters
    ----------
    V : (n,d) numpy array or None
        vertex list of a mesh.
        If this is None, will assume a mesh with max(F)+1 vertices.
    F : (m,k) numpy int array
        index list of a polyline/triangle mesh/tet mesh
    return_maps : bool, optional (default False)
        If True, will return mapping arrays `I` and `J` that can be used to map
        between the new and old vertex indices.

    Returns
    -------
    newV : (n1,d) numpy array or None
           vertex list with unreferenced vertices removed, or None if V is None.
    newF : (m1,d) numpy int array
           index list with unreferenced vertices removed
    I : (n,) or (n+1,) numpy int array, if requested
        index list such that `I[F] == newF`.
        `I` is of length `(n,)` if `F` does not contain `-1`, and of length
        `(n+1,)` if `F` does contain `-1`.
    J : (n1,) numpy int array, if requested
        index list such that `V[J,:] = newV`.

    See Also
    --------
    libigl's remove_unreferenced, which this function is inspired by, but does
    not perfectly mirror (in the presence of `-1` in the face list, `-1` will be
    appended to the return value `I` to ensure that `I(F) == newF`).

    Examples
    --------
    ```python
    >>> V = np.array([[0.,0.],[1.,0.],[0.,1.],[-1.,-2.],[1.,1.]])
    >>> F = np.array([[0,1,2],[2,1,4]])
    >>> newV, newF, I, J = gpy.remove_unreferenced(V, F, return_maps=True)
    >>> newV
    array([[0., 0.],
           [1., 0.],
           [0., 1.],
           [1., 1.]])
    >>> newF
    array([[0, 1, 2],
           [2, 1, 3]])
    >>> I
    array([ 0,  1,  2, -1,  3])
    >>> J
    array([0, 1, 2, 4])
    ```
    
    """

    assert (F>=-1).all(), "The only allowed negative value is -1."
    if V is None:
        n = np.max(F)+1
    else:
        n = V.shape[0]
        assert np.max(F)<n, "V is too small for F."

    J,idx,newF = np.unique(F, return_index=True, return_inverse=True)
    if len(J)<1:
        emptyV = None if V is None else np.array([],dtype=V.dtype)
        emptyF = np.array([],dtype=F.dtype)
        if return_maps:
            return emptyV, emptyF, emptyF.copy(), emptyF.copy()
        else:
            return emptyV, emptyF
    if J[0] == -1:
        J = J[1:]
        newF -= 1
        I = np.full((n+1,), -1)
    else:
        I = np.full((n,), -1)
    I[J] = np.arange(len(J))
    newF = newF.reshape(F.shape)
    newV = None if V is None else V[J,:]
    
    if return_maps:
        return newV, newF, I, J
    else:
        return newV, newF


