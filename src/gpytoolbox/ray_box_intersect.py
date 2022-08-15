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

    Examples
    --------
    
    """
    
    ##Legend: (RIGHT	0) (LEFT	1) (MIDDLE     2)
# char HitBoundingBox(minB,maxB, origin, dir,coord)
# double minB[NUMDIM], maxB[NUMDIM];		/*box */
# double origin[NUMDIM], dir[NUMDIM];		/*ray */
# double coord[NUMDIM];				/* hit point */
# {
# 	char inside = TRUE;
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
# 	char quadrant[NUMDIM];
# 	register int i;
# 	int whichPlane;
# 	double maxT[NUMDIM];
# 	double candidatePlane[NUMDIM];
    for i in range(dim):
# 	/* Find candidate planes; this loop can be avoided if
#    	rays cast all from the eye(assume perpsective view) */
# 	for (i=0; i<NUMDIM; i++)
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

# 		if(origin[i] < minB[i]) {
# 			quadrant[i] = LEFT;
# 			candidatePlane[i] = minB[i];
# 			inside = FALSE;
# 		}else if (origin[i] > maxB[i]) {
# 			quadrant[i] = RIGHT;
# 			candidatePlane[i] = maxB[i];
# 			inside = FALSE;
# 		}else	{
# 			quadrant[i] = MIDDLE;
# 		}

    
    if inside:
        hit_coord = origin
        return True, hit_coord

# 	/* Ray origin inside bounding box */
# 	if(inside)	{
# 		coord = origin;
# 		return (TRUE);
# 	}

    for i in range(dim):
        if ( (quadrant[i]!=2) and  (dir[i]!=0.) ):
            maxT[i] = (candidatePlane[i]-origin[i]) / dir[i]
        else:
            maxT[i] = -1.

# 	/* Calculate T distances to candidate planes */
# 	for (i = 0; i < NUMDIM; i++)
# 		if (quadrant[i] != MIDDLE && dir[i] !=0.)
# 			maxT[i] = (candidatePlane[i]-origin[i]) / dir[i];
# 		else
# 			maxT[i] = -1.;
    whichPlane = 0
    for i in range(dim):
        if (maxT[whichPlane] < maxT[i]):
            whichPlane = i

# 	/* Get largest of the maxT's for final choice of intersection */
# 	whichPlane = 0;
# 	for (i = 1; i < NUMDIM; i++)
# 		if (maxT[whichPlane] < maxT[i])
# 			whichPlane = i;

    

# 	/* Check final candidate actually inside box */
    if (maxT[whichPlane] < 0.):
        return False, hit_coord
# 	if (maxT[whichPlane] < 0.) return (FALSE);

    for i in range(dim):
        if (whichPlane != i):
            hit_coord[i] = origin[i] + maxT[whichPlane]*dir[i]
            if ( (hit_coord[i] < minB[i]) or (hit_coord[i] > maxB[i])):
                return False, hit_coord
        else:
            hit_coord[i] = candidatePlane[i]
    
    return True, hit_coord
        
# 	for (i = 0; i < NUMDIM; i++)
# 		if (whichPlane != i) {
# 			coord[i] = origin[i] + maxT[whichPlane] *dir[i];
# 			if (coord[i] < minB[i] || coord[i] > maxB[i])
# 				return (FALSE);
# 		} else {
# 			coord[i] = candidatePlane[i];
# 		}
# 	return (TRUE);				/* ray hits box */
# }	