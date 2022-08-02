import numpy as np
from scipy.sparse import csr_matrix
from .per_face_normals import per_face_normals
from .doublearea import doublearea


def per_vertex_normals(V,F):
    """Normal vectors to all vertices on a mesh
    
    Computs area-weighted per-vertex unit normal vectors for a triangle mesh

    Parameters
    ----------
    V : numpy double array
        Matrix of mesh vertex coordinates
    F : numpy int array
        Matrix of triangle indices into V

    Returns
    -------
    N : numpy double array
        Matrix of per-vertex normals

    See Also
    --------
    per_face_normals.

    Examples
    --------
    TODO
    """
    # Computs per vertex unit normal vectors for a triangle mesh
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       N #F by 3 numpy array of per-vertex unit normals
    #           


    # First compute face normals
    face_normals = per_face_normals(V,F,unit_norm=True)
    # We blur these normals onto vertices, weighing by area
    areas = doublearea(V,F)
    vals = np.concatenate((areas,areas,areas))
    J = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)
    J = np.concatenate((J,J,J))
    I = np.concatenate((F[:,0],F[:,1],F[:,2]))

    weight_mat = csr_matrix((vals,(I,J)),shape=(V.shape[0],F.shape[0]))

    vertex_normals = weight_mat @ face_normals
    # Now, normalize
    N = vertex_normals/np.tile(np.linalg.norm(vertex_normals,axis=1)[:,None],(1,3))

    return N
