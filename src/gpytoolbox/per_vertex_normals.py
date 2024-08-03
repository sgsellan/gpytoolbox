import numpy as np
from .per_face_normals import per_face_normals
from .doublearea import doublearea
from scipy.sparse import csr_matrix


def compute_angles(V,F):
    """Computes the angle at each vertex in a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh 
    F : (m,d) numpy int array
        face index list of a triangle mesh 

    Returns
    -------
    angles : (m,d) numpy array
        Array containing the angles at each vertex of each triangle in radians

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, compute_angles
    v,f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    angles = compute_angles(v,f)
    ```

    Notes
    -----
    The angles are computed using the dot product between vectors formed by 
    the vertices of each triangle. The angles are then calculated using the 
    arccosine of the dot product result, which is safeguarded against numerical 
    issues by clipping the values to the range [-1, 1].
    """
    # Extract the vertices of each triangle
    A = V[F[:, 0]]
    B = V[F[:, 1]]
    C = V[F[:, 2]]

    # Compute the edge vectors for each triangle
    AB = B - A
    AC = C - A
    BC = C - B
    CA = A - C
    BA = A - B
    CB = B - C

    # Cosines via dot product
    cos_a = np.einsum('ij,ij->i', (B - A), (C - A)) / (np.linalg.norm(AB, axis=1) * np.linalg.norm(AC, axis=1))
    cos_b = np.einsum('ij,ij->i', (C - B), (A - B)) / (np.linalg.norm(BC, axis=1) * np.linalg.norm(BA, axis=1))
    cos_c = np.einsum('ij,ij->i', (A - C), (B - C)) / (np.linalg.norm(CA, axis=1) * np.linalg.norm(CB, axis=1))

    # Angles via arccos, safeguarded against numerical issues via clipping
    angles = np.zeros((F.shape[0], 3))
    angles[:, 0] = np.arccos(np.clip(cos_a, -1, 1))
    angles[:, 1] = np.arccos(np.clip(cos_b, -1, 1))
    angles[:, 2] = np.arccos(np.clip(cos_c, -1, 1))

    return angles


def per_vertex_normals(V,F,weights='area'):
    """Compute per-vertex normal vectors for a triangle mesh with different weighting options.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,d) numpy int array
        face index list of a triangle mesh
    weights : str, optional
        The type of weighting to use ('area', 'angles', 'uniform'), default is 'area'

    Returns
    -------
    N : (n,d) numpy array
        Matrix of per-vertex normals

    See Also
    --------
    per_vertex_normals, per_face_normals.

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, per_vertex_normals
    v,f = read_mesh("test/unit_tests_data/armadillo.obj")
    n = per_vertex_normals(v,f)
    ```
    """
    if weights=="area":
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


    elif weights=="angles":
        # Ensure vertices are in float64
        V = V.astype(np.float64)  
        # Ensure faces are in int32
        F = F.astype(np.int32)    
        
        # Compute face normals
        normals = per_face_normals(V,F)
        # Compute angles at each vertex
        angles = compute_angles(V,F)
        # Weight the face normals by the angles at each vertex
        weighted_normals = np.zeros_like(V)
        for i in range(3):
            np.add.at(weighted_normals, F[:, i], normals * angles[:, i, np.newaxis])
        
        # Calculate norms
        norms = np.linalg.norm(weighted_normals, axis=1, keepdims=True)
        # Avoid division by zero by setting zero norms to a very small number
        norms[norms == 0] = np.finfo(float).eps
        # Normalize the vertex normals
        N = weighted_normals / norms
    

    elif weights=="uniform":
        # Compute face normals
        face_normals = per_face_normals(V, F)
        # Initialize vertex normals
        vertex_normals = np.zeros_like(V)
        # Accumulate face normals to vertices
        for i in range(3):
            np.add.at(vertex_normals, F[:, i], face_normals)
        
        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = np.finfo(float).eps
        N = vertex_normals / norms

    return N