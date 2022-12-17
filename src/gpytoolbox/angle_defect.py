import numpy as np
from .halfedge_lengths_squared import halfedge_lengths_squared
from .angle_defect_intrinsic import angle_defect_intrinsic

def angle_defect(V,F,
    use_small_angle_approx=True):
    """Returns the angle defect at each vertex (except at boundary vertices,
    where it is `0`.).
    The angle defect is a proxy for integrated Gaussian curvature.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    use_small_angle_approx : bool, optional (default True)
        If True, uses a different, more more stable formula for small angles.

    Returns
    -------
    k : (n,) numpy array
        angle defect at each vertex.

    Examples
    --------
    ```python
    V,F = gpy.read_obj("mesh.obj")
    k = gpy.angle_defect(V,F)
    ```
    """

    l_sq = halfedge_lengths_squared(V,F)
    return angle_defect_intrinsic(l_sq,F,n=V.shape[0],
        use_small_angle_approx=use_small_angle_approx)
