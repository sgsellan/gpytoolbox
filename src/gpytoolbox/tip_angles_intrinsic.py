import numpy as np

def tip_angles_intrinsic(l_sq, F, use_small_angle_approx=True):
    # Computes the angles formed by each vertex within its respective face
    # (the tip angle) using only intrinsic information (squared halfedge edge
    # lengths).
    #
    # Input:
    #       l_sq  #F by 3 numpy array of squared halfedge lengths as computed
    #             by halfedge_lengths_squared
    #       F  #F by 3 int numpy array of face/edge vertex indeces into V
    #       Optional:
    #                 use_small_angle_approx  if True, uses a different, more
    #                                         more stable formula for small
    #                                         angles.
    #
    # Output:
    #       alpha  #F by 3 numpy array of tip angles for each vertex

    assert F.shape[1] == 3
    assert l_sq.shape == F.shape
    assert np.all(l_sq>=0)

    #Using cosine rule
    def cr(a, b, c):
        cosgamma = (a+b-c) / (2.*np.sqrt(a*b))
        gamma = np.arccos(np.clip(cosgamma, -1., 1.))
        if use_small_angle_approx:
            #If c is very small, expect numerical error and use a series approx.
            small = np.sqrt(np.finfo(c.dtype).eps) #This is quite aggressive, can also try without sqrt
            i = c<small
            if np.any(i):
                # https://www.nayuki.io/page/numerically-stable-law-of-cosines
                sqrt_ai, sqrt_bi = np.sqrt(a[i]), np.sqrt(b[i])
                gsq = (c[i] - (sqrt_ai-sqrt_bi)**2) / (sqrt_ai*sqrt_bi)
                gamma[i] = np.sqrt(np.clip(gsq, 0., None))
        return gamma
    alpha = np.stack(
        [cr(l_sq[:,1], l_sq[:,2], l_sq[:,0]),
        cr(l_sq[:,2], l_sq[:,0], l_sq[:,1]),
        cr(l_sq[:,0], l_sq[:,1], l_sq[:,2])],
        axis=-1)

    return alpha
    