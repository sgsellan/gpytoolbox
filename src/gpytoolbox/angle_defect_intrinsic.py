import numpy as np
from .tip_angles_intrinsic import tip_angles_intrinsic
from .boundary_vertices import boundary_vertices

def angle_defect_intrinsic(l_sq,F,n=None,
    use_small_angle_approx=True):
    """Returns the angle defect at each vertex (except at boundary vertices,
    where it is `0`.).
    The angle defect is a proxy for integrated Gaussian curvature.
    This function uses only intrinsic information (halfedge lengths squared)

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh
    n : int, optional (default None)
        number of vertices in the mesh.
        If absent, will try to infer from F.
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
    l_sq = gpy.halfedge_lengths_squared(V,F)
    k = gpy.angle_defect_intrinsic(l_sq,F)
    ```
    """

    assert F.shape[1] == 3
    assert l_sq.shape == F.shape
    assert np.all(l_sq>=0)

    if n==None:
        n = 0

    α = tip_angles_intrinsic(l_sq,F,
        use_small_angle_approx=use_small_angle_approx)
    α_sum = np.bincount(F.ravel(), α.ravel(), minlength=n)

    k = 2.*np.pi - α_sum
    b = boundary_vertices(F)
    k[b] = 0.

    return k
