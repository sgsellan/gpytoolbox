import numpy as np

def remove_duplicate_vertices(V,epsilon=0.0,faces=None):
    """Return unique vertices, optionally with a tolerance
    
    Parameters
    ----------
    V : numpy double array
        Matrix of vertex positions
    epsilon : double, optional (default 0.0)
        Positive uniqueness absolute tolerance
    faces : numpy int array, optional (default None)
        Matrix of any-type mesh indices, for convenience
    
    Returns
    -------
    SV : numpy double array
        Matrix of new vertex positions
    SVI : numpy int array
        Vector of indices such that SV = V[SVI,:]
    SVJ : numpy int array
        Vector of indices such that V = SV[SVJ,:]
    SF : numpy int array
        Matrix of new mesh indices into SV, only part of the output if faces is not None.

    See Also
    --------
    remove_unreferenced.

    Examples
    --------
    ```python
    # Build a polyline; for example, a square, but duplicate the last point
    V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0] ])
    # Remove duplicate vertices
    SV, SVI, SVJ = gpytoolbox.remove_duplicate_vertices(V)
    # SV is now unique
    ```
    """
    
    if epsilon==0.0:
        SV, SVI, SVJ = np.unique(V,return_index=True,return_inverse=True,axis=0)
    else:
        _, SVI, SVJ = np.unique(np.round(V/epsilon),return_index=True,return_inverse=True,axis=0)
        SV = V[SVI,:]

    if (faces is None):
        return SV, SVI, SVJ
    else:
        return SV, SVI, SVJ, SVJ[faces]