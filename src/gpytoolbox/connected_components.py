import numpy as np
import scipy as sp

from .adjacency_matrix import adjacency_matrix

def connected_components(F,
    return_face_indices=False):
    """Computes the connected components of a triangle mesh.
    This follows the convention of gptoolbox (https://github.com/alecjacobson/gptoolbox/blob/master/mesh/connected_components.m)

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh
    return_face_indices : bool, optional (default False)
        whether to return component IDs for faces for faces

    Returns
    -------
    C : (n,) numpy int array
        connected component ID for each vertex
    CF : if requested, (m,) numpy int array
        connected component ID for each face

    Examples
    --------
    ```python
    V,F = gpy.read_mesh("mesh.obj")
    C,CF = gpy.connected_components(F)
    ```
    """

    if F.size==0:
        C = np.array([], dtype=int)
        CF = np.array([], dtype=int)
    else:
        assert F.shape[1]==3, "This function only works for triangle meshes."

        A = adjacency_matrix(F)
        _,C = sp.sparse.csgraph.connected_components(A)
        CF = C[F[:,0]]

    if return_face_indices:
        return C,CF
    else:
        return C
