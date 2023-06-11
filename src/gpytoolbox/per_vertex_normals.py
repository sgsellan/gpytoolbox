import numpy as np
from scipy.sparse import csr_matrix
from .per_face_normals import per_face_normals
from .doublearea import doublearea


def per_vertex_normals(V,F):
    """Normal vectors to all vertices on a mesh
    
    Computes area-weighted per-vertex unit normal vectors for a triangle mesh or polyline.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh or polyline
    F : (m,d) numpy int array
        face index list of a triangle mesh (edge list for a polyline)

    Returns
    -------
    N : (m,d) numpy double array
        Matrix of per-vertex normals

    See Also
    --------
    per_face_normals.

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, per_vertex_normals
    v,f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    n = per_vertex_normals(v,f)
    ```
    """
    dim = V.shape[1]
    # First compute face normals
    face_normals = per_face_normals(V,F,unit_norm=True)
    # We blur these normals onto vertices, weighing by area
    areas = doublearea(V,F)
    vals = np.concatenate([areas for _ in range(dim)])
    J = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)
    # J = np.concatenate([J,J,J))
    J = np.concatenate([J for _ in range(dim)])
    I = np.concatenate([F[:,dd] for dd in range(dim)])
    # I = np.concatenate((F[:,0],F[:,1],F[:,2]))

    weight_mat = csr_matrix((vals,(I,J)),shape=(V.shape[0],F.shape[0]))

    vertex_normals = weight_mat @ face_normals
    # Now, normalize
    N = vertex_normals/np.tile(np.linalg.norm(vertex_normals,axis=1)[:,None],(1,dim))

    return N
