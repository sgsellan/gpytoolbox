import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.tip_angles_intrinsic import tip_angles_intrinsic

def tip_angles(V, F, use_small_angle_approx=True):
    # Computes the angles formed by each vertex within its respective face
    # (the tip angle).
    #
    # Input:
    #       V  #V by 3 numpy array of mesh vertex positions
    #       F  #F by 3 int numpy array of face/edge vertex indeces into V
    #       Optional:
    #                 use_small_angle_approx  if True, uses a different, more
    #                                         more stable formula for small
    #                                         angles.
    #
    # Output:
    #       alpha  #F by 3 numpy array of tip angles for each vertex

    l_sq = halfedge_lengths_squared(V,F)
    return tip_angles_intrinsic(l_sq,F,use_small_angle_approx)
