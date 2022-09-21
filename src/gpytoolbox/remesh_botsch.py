import numpy as np
from gpytoolbox.boundary_vertices import boundary_vertices
from gpytoolbox.halfedge_lengths import halfedge_lengths

def remesh_botsch(V,F,i=10,h=None,project=True,feature = np.array([],dtype=int)):
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
    feature : numpy int array, optional (default np.array([],dtype=int))
        List of indeces of feature vertices that should not change (i.e., they will also be in the output) 
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
    ```python
    # Read a mesh
    v,f = gpytoolbox.read_mesh("bunny_oded.obj")
    # Do 20 iterations of remeshing with a target length of 0.01
    u,g = gpytoolbox.remesh_botsch(v,f,20,0.01,True)
    ```
    """
    try:
        from gpytoolbox_bindings import _remesh_botsch_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")
    
    if (h is None):
        h = np.mean(halfedge_lengths(V,F))

    feature = np.concatenate((feature,boundary_vertices(F)),dtype=np.int32)
    # print(feature)
    # print(boundary_vertices(F))
    # bV = boundary_vertices(F)


    v,f = _remesh_botsch_cpp_impl(V,F.astype(np.int32),i,h,feature,project)

    return v,f
