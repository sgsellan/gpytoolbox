import numpy as np

from gpytoolbox.halfedge_lengths import halfedge_lengths

def remesh_botsch(V,F,i=10,h=None,project=True):
    """Remesh a triangular mesh to have a desired edge length
    
    Use the algorithm described by Botsch and Kobbelt's "A Remeshing Approach to Multiresolution Modeling" to remesh a triangular mesh by alternating iterations of subdivision, collapse, edge flips and collapses.

    Parameters
    ----------
    V : numpy double array
        Matrix of triangle mesh vertex coordinates
    F : numpy int array
        Matrix of triangle mesh indices into V
    i : int, optional (default 10)
        Number of algorithm iterations
    h : double, optional (default 0.1)
        Desired edge length (if None, will pick average edge length)
    project : bool, optional (default True)
        Whether to reproject the mesh to the input (otherwise, it will smooth over iterations).

    Returns
    -------
    U : numpy double array
        Matrix of output triangle mesh vertex coordinates
    G : numpy int array
        Matrix of output triangle mesh indices into U

    Examples
    --------
    TODO
    """
    try:
        from gpytoolbox_bindings import _remesh_botsch_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")
    
    if (h is None):
        h = np.mean(halfedge_lengths(V,F))

    v,f = _remesh_botsch_cpp_impl(V,F.astype(np.int32),i,h,project)

    return v,f
