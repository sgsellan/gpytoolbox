import numpy as np



def per_face_normals(V,F,unit_norm=True):
    """Vector perpedicular to all faces on a mesh
    
    Computes per face (optionally unit) normal vectors for a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    unit_norm : bool, optional (default True)
        Whether to normalize each face's normal before outputting

    Returns
    -------
    N : (n,d) numpy double array
        Matrix of per-face normals

    See Also
    --------
    per_vertex_normals.

    Examples
    --------
    TODO
    """
    # Computes per face (optionally unit) normal vectors for a triangle mesh
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #       Optional:
    #               'unit_norm' boolean, whether to normalize each face's 
    #                       normal before outputting {default: true}
    #
    # Output:
    #       N #F by 3 numpy array of per-face normals
    #           

    v0 = V[F[:,0],:]
    v1 = V[F[:,1],:]
    v2 = V[F[:,2],:]

    # It's basically just a cross product
    N = np.cross(v1-v0,v2-v0,axis=1)

    if unit_norm:
        N = N/np.tile(np.linalg.norm(N,axis=1)[:,None],(1,3))

    return N
