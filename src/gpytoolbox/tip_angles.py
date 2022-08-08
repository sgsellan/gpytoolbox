import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.tip_angles_intrinsic import tip_angles_intrinsic

def tip_angles(V, F,
    use_small_angle_approx=True):
    """Computes the angles formed by each vertex within its respective face
    (the tip angle).

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    use_small_angle_approx : bool, optional (default: True)
        If True, uses a different, more more stable formula for small angles.

    Returns
    -------
    α : (m,3) numpy array
        tip angles for each vertex referenced in `F`

    Examples
    --------
    TODO
    
    """

    l_sq = halfedge_lengths_squared(V,F)
    return tip_angles_intrinsic(l_sq,F,use_small_angle_approx)
