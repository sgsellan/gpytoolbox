import numpy as np

def ray_triangle_intersect(origin,dir, v0, v1, v2, return_negative=False):
    """Find if and where a straight line intersects a triangle in 3D

    Parameters
    ----------
    origin : (dim,) numpy double array
        Ray origin position coordinates
    dir : (dim,) numpy double array
        Ray origin direction coordinates
    v0 : (dim,) numpy double array
        First triangle vertex coordinates
    v1 : (dim,) numpy double array
        Second triangle vertex coordinates
    v2 : (dim,) numpy double array
        Third triangle vertex coordinates
    return_negative : bool, optional (default False)
        Whether to return intersections at negative time (i.e., "behind" the ray)
    
    Returns
    -------
    t : double
        "Time" it takes the ray to hit the triangle (may be negative)
    is_hit : bool
        Whether the ray ever hits the triangle
    hit_coord : (dim,) numpy double array
        Intersection coordinates. Some entries of this vector will be infinite if is_hit is False.

    See Also
    --------
    ray_polyline_intersect, ray_mesh_intersect.

    Notes
    -----
    Uses the Möller–Trumbore ray-triangle intersection algorithm.

    Examples
    --------
    ```python
    from gpytoolbox import ray_triangle_intersect
    # Random origin and direction
    origin = np.random.rand(3)
    dir = np.random.rand(3)
    # Random vertices
    V = np.random.rand(3,3)
    v1 = V[0,:]
    v2 = V[1,:]
    v3 = V[2,:]
    t,is_hit,hit_coord = ay_triangle_intersect(origin,dir,v1,v2,v3)
    ```
    """
    v0 = np.ravel(v0)
    v1 = np.ravel(v1)
    v2 = np.ravel(v2)
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(dir,v0v2)
    det = np.dot(v0v1,pvec)

    if (np.abs(det) < 1e-6):
        t =  np.Inf
        hit = np.Inf*np.ones(3)
        is_hit = False
        return t,is_hit, hit

    invDet = 1.0 / det
    tvec = origin - v0
    u = np.dot(tvec,pvec) * invDet

    if (u < 0 or u > 1):
        t =  np.Inf
        hit = np.Inf*np.ones(3)
        is_hit = False
        return t,is_hit, hit

    qvec = np.cross(tvec,v0v1)
    v = np.dot(dir,qvec)*invDet

    if ((v < 0) or ((u + v) > 1)):
        t =  np.Inf
        hit = np.Inf*np.ones(3)
        is_hit = False
        return t,is_hit, hit

    t = np.dot(v0v2,qvec)*invDet
    hit = origin + t*dir
    is_hit = True
    if (not return_negative):
        if(t<0):
            t =  np.Inf
            hit = np.Inf*np.ones(3)
            is_hit = False
    return t,is_hit, hit


    