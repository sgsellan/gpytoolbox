import numpy as np
from .per_face_normals import per_face_normals
from .doublearea import doublearea
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from .tip_angles import tip_angles

def compute_angles(V, F):
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


def per_vertex_normals(V, F, weights='area', correct_invalid_normals=True):
    """Compute per-vertex normal vectors for a triangle mesh with different weighting options,
    with an option to correct invalid normals (True) or omit them (False).

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,d) numpy int array
        face index list of a triangle mesh
    weights : str, optional
        the type of weighting to use ('area', 'angle', 'uniform'), default is 'area'
    correct_invalid_normals : bool, optional
        if True, invalid normals (NaN, inf, zero vectors) will be corrected, either by calculating
        them based on nearby valid normals, or -in extreme cases- replacing them with a default normal vector
        if False, the invalid normals are omitted

    Returns
    -------
    N : (n,d) numpy array
        matrix of per-vertex normals

    See Also
    --------
    per_face_normals

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, per_vertex_normals
    v,f = read_mesh("test/unit_tests_data/armadillo.obj")
    n = per_vertex_normals(v,f)
    ```
    """
    # Ensure vertices are in float64
    V = V.astype(np.float64)  
    # Ensure faces are in int32
    F = F.astype(np.int32)  

    # Compute face normals
    face_normals = per_face_normals(V, F, unit_norm=True)

    if weights=="area":
        dim = V.shape[1]
        # We blur these normals onto vertices, weighing by area
        areas = doublearea(V,F)
        vals = np.concatenate([areas for _ in range(dim)])
        J = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)
        # J = np.concatenate([J,J,J))
        J = np.concatenate([J for _ in range(dim)])
        I = np.concatenate([F[:,dd] for dd in range(dim)])
        # I = np.concatenate((F[:,0],F[:,1],F[:,2]))

        weight_mat = csr_matrix((vals,(I,J)),shape=(V.shape[0],F.shape[0]))
        weighted_normals = weight_mat @ face_normals
       
    elif weights=="angle":
        # Compute angles at each vertex
        angles = tip_angles(V, F)
        # Weight the face normals by the angles at each vertex
        weighted_normals = np.zeros_like(V)
        for i in range(3):
            np.add.at(weighted_normals, F[:, i], face_normals * angles[:, i, np.newaxis])
   
    elif weights=="uniform":
        # Compute (non-)weighted normals uniformly
        weighted_normals = np.zeros_like(V)
        # Accumulate face normals to vertices
        for i in range(3):
            np.add.at(weighted_normals, F[:, i], face_normals)
            
    # Calculate norms
    norms = np.linalg.norm(weighted_normals, axis=1, keepdims=True)
    # Identify indices with problematic normals (nan, inf, negative norms)
    problematic_indices = np.where(np.isnan(norms) | np.isinf(norms) | (norms <= 0))[0]
    # Identify indices with valid normals
    valid_indices = np.where(np.isfinite(norms) & (norms > 0))[0]

    # Normalize valid normals
    N = np.where(norms > 0, weighted_normals / norms, np.zeros_like(weighted_normals))

    # Ensure all normals are unit vectors (handling edge cases)
    if problematic_indices.size > 0 and correct_invalid_normals:
        print("Handling problematic indices")
        
        # Build KDTree using only valid vertices
        if valid_indices.size > 0:
            valid_tree = KDTree(V[valid_indices])

        for idx in problematic_indices:
            print(f"Handling vertex {idx}")

            # Use nearest valid normal for invalid normals
            if valid_indices.size > 0:
                # Query the nearest valid normal
                dist, pos = valid_tree.query(V[idx], k=1)  
                nearest_valid_idx = valid_indices[pos]  
                # Assign the nearest valid normal to the current problematic normal
                if np.isfinite(dist) and dist > 0:
                    N[idx] = N[nearest_valid_idx]
                    norms[idx] = np.linalg.norm(N[nearest_valid_idx], keepdims=True)
                    print(f"Updated normal for vertex {idx} from valid neighbor.")
                # Else assign a default normal
                else:
                    N[idx] = np.array([1, 0, 0])
                    norms[idx] = 1.0
                    print(f"Assigned default normal due to lack of valid neighbors.")
            
            # Else assign a default normal
            else:
                N[idx] = np.array([1, 0, 0])
                norms[idx] = 1.0
                print(f"No valid vertices available, default normal assigned.")

        # Normalize only the previously problematic normals
        N[problematic_indices] = N[problematic_indices] / np.maximum(norms[problematic_indices], np.finfo(float).eps)

    return N