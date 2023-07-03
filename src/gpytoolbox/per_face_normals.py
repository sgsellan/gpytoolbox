import numpy as np



def per_face_normals(V,F,unit_norm=True):
    """Vector perpedicular to all faces on a mesh
    
    Computes per face (optionally unit) normal vectors for a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,d) numpy int array
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
    ```python
    from gpytoolbox import read_mesh, per_face_normals
    v,f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    n = per_face_normals(v,f,unit_norm=True)
    ```
    """     

    dim = V.shape[1]


    if dim == 2:
        # Edge vectors
        v0 = V[F[:,0],:]
        v1 = V[F[:,1],:]
        # Difference between edge vectors
        e = v1-v0
        # Rotate by 90 degrees
        N = np.hstack((e[:,1][:,None],-e[:,0][:,None]))
        # print(N)
    elif dim == 3:     
        v0 = V[F[:,0],:]
        v1 = V[F[:,1],:]
        v2 = V[F[:,2],:]

        # It's basically just a cross product
        N = np.cross(v1-v0,v2-v0,axis=1)

    if unit_norm:
        N = N/np.tile(np.linalg.norm(N,axis=1)[:,None],(1,dim))

    return N
