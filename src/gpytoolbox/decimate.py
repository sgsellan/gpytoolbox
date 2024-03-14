import numpy as np

def decimate(V,F,
    face_ratio=0.1,
    num_faces=None,
    method='shortest_edge'):
    """Reduce the number of faces of a triangle mesh.

    From a manifold triangle mesh, builds a new triangle mesh with fewer faces than the original one using libigl's decimation algorithm.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    face_ratio: double, optional (default 0.1)
        Desired ratio of output faces to input faces 
    num_faces : int, optional (default None)
        Desired number of faces in output mesh (superseeds face_ratio if set)
    method : string, optional (default shortest_edge)
        Which mesh decimation algorithm to use.
        Options are 'shortest_edge' and 'qslim'

    Returns
    -------
    U : numpy double array
        Matrix of mesh vertices
    G : numpy int array
        Matrix of triangle indices
    I : numpy int array
        Vector of indices into F of the original faces in G
    J : numpy int array
        Vector of indices into V of the original vertices in U

    See Also
    --------
    lazy_cage.

    Notes
    -----
    As described in libigl, this collapses the shortest edges first, placing the new vertex at the edge midpoint, and stops when the desired number of faces is reached or no face can be collapsed without going below the desired number of faces.

    Examples
    --------
    ```python
    # Mesh in v,f
    u,g,i,j = gpytoolbox.decimate(v,f,num_faces=100)
    # New mesh in u,g
    ```
    """

    # Try to import C++ binding
    try:
        from gpytoolbox_bindings import _decimate_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ decimate binding.")

    if (num_faces is None):
        num_faces = np.floor(face_ratio*F.shape[0]).astype(np.int32)

    method_int = 0
    if method == 'shortest_edge':
        method_int = 0
    elif method == 'qslim':
        method_int = 1
    else:
        raise Exception("Not a valid decimation method.")
    v, f, i, j = _decimate_cpp_impl(V.astype(np.float64),F.astype(np.int32),
        num_faces,
        method_int)

    return v,f,i,j
