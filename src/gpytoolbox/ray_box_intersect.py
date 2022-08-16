import numpy as np

def ray_box_intersect(origin,dir,center,width):
    """Find if and where a straight line intersects an axis-aligned box

    Parameters
    ----------
    origin : (dim,) numpy double array
        Ray origin position coordinates
    dir : (dim,) numpy double array
        Ray origin direction coordinates
    center : (dim,) numpy double array
        Box center coordinates
    width : (dim,) numpy double array
        Box widths in each dimension
    
    Returns
    -------
    is_hit : bool
        Whether the ray ever hits the box
    hit_coord : (dim,) numpy double array
        Intersection coordinates (one of them, there may be many). Some entries of this vector will be infinite if is_hit is False.

    See Also
    --------
    ray_polyline_intersect, ray_mesh_intersect.

    Notes
    -----
    Uses the algorithm described in "Graphics Gems" by Eric Haines. Note that this will check intersections through the whole straight line, e.g., for positive and negative time values.

    Examples
    --------
    ```python
    from gpytoolbox import ray_box_intersect
    center = np.array([0,0])
    width = np.array([1,1])
    position = np.array([-1,0])
    dir = np.array([1.0,0.1])
    is_hit,where_hit = gpytoolbox.ray_box_intersect(position,dir,center,width)
    ```
    """
    assert(np.ndim(origin)==1)
    dim = origin.shape[0]
    inside = True
    quadrant = -np.ones(dim)
    maxT = -np.ones(dim)
    candidatePlane = -np.ones(dim)
    whichPlane = 0
    minB = center - 0.5*width
    maxB = center + 0.5*width
    hit_coord = np.Inf*np.ones(dim)

    for i in range(dim):
        if (origin[i]<minB[i]):
            quadrant[i] = 1
            candidatePlane[i] = minB[i]
            inside = False
        elif (origin[i]>maxB[i]):
            quadrant[i] = 0
            candidatePlane[i] = maxB[i]
            inside = False
        else:
            quadrant[i] = 2

    
    if inside:
        hit_coord = origin
        return True, hit_coord

    for i in range(dim):
        if ( (quadrant[i]!=2) and  (dir[i]!=0.) ):
            maxT[i] = (candidatePlane[i]-origin[i]) / dir[i]
        else:
            maxT[i] = -1.

    whichPlane = 0
    for i in range(dim):
        if (maxT[whichPlane] < maxT[i]):
            whichPlane = i

    if (maxT[whichPlane] < 0.):
        return False, hit_coord

    for i in range(dim):
        if (whichPlane != i):
            hit_coord[i] = origin[i] + maxT[whichPlane]*dir[i]
            if ( (hit_coord[i] < minB[i]) or (hit_coord[i] > maxB[i])):
                return False, hit_coord
        else:
            hit_coord[i] = candidatePlane[i]
    
    return True, hit_coord