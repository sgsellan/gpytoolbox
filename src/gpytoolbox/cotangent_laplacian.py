import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.cotangent_laplacian_intrinsic import cotangent_laplacian_intrinsic

def cotangent_laplacian(V,F):
    """Builds the (pos. def.) cotangent Laplacian for a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    L : (n,n) scipy csr_matrix
        cotangent Laplacian

    Examples
    --------
    ```python
    # Mesh in V,F
    from gpytoolbox import cotangent_laplacian
    L = cotangent_laplacian(V,F)
    ```
    """

    l_sq = halfedge_lengths_squared(V,F)
    return cotangent_laplacian_intrinsic(l_sq,F,n=V.shape[0])
