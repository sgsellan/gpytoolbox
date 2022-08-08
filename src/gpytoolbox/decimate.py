import numpy as np

def decimate(V,F,face_ratio=0.1,num_faces=None):
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

    Returns
    -------
    U : numpy double array
        Matrix of mesh vertices
    G : numpy int array
        Matrix of triangle indices
    I : numpy int array
        Vector of indeces into F of the original faces in G
    J : numpy int array
        Vector of indeces into V of the original vertices in U

    See Also
    --------
    lazy_cage.

    Notes
    -----
    As described in libigl, this collapses the shortest edges first, placing the new vertex at the edge midpoint, and stops when the desired number of faces is reached or no face can be collapsed without going below the desired number of faces.

    Examples
    --------
    TO-DO
    """

    # Try to import C++ binding
    try:
        from gpytoolbox_bindings import _decimate_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ decimate binding.")

    if (num_faces is None):
        num_faces = np.floor(face_ratio*F.shape[0]).astype(np.int32)

    v, f, i, j = _decimate_cpp_impl(V,F.astype(np.int32),num_faces)

    return v,f,i,j
