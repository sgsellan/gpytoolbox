import numpy as np



def per_face_normals(V,F,unit_norm=True):
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
