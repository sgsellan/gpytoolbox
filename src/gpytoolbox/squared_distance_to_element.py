import numpy as np
from gpytoolbox.barycentric_coordinates import barycentric_coordinates

def squared_distance_to_element(point,V,element):
    """Squared distance from a point to a mesh element (point, edge, triangle)

    Parameters
    ----------
    point : (dim,) numpy double array
        Query point coordinates
    V : (v,dim) numpy double array
        Matrix of mesh/polyline/pointcloud vertex coordinates
    element : (s,) numpy int array
        Vector of element indices into V

    Returns
    -------
    sqrD : double
        Squared minimum distance from point to mesh element

    See Also
    --------
    squared_distance.

    Examples
    --------
    ```python
    # Generate random mesh
    V = np.random.rand(3,3)
    F = np.array([0,1,2],dtype=int)
    # Generate random query point
    P = np.random.rand(3)
    # Calculate distance from point to triangle
    sqrD = gpytoolbox.squared_distance_to_element(P,V,F)
    ```
    """
    dim = V.shape[1]
    if element.ndim>1:
        element = np.ravel(element)
    simplex_size = element.shape[0]
    if simplex_size==1:
        # Then this is just distance between two points
        sqrD = np.sum((V[element] - point)**2.0)
        lmb = 1
    elif simplex_size==2:
        if dim==2:
            V = np.hstack(( V,np.zeros((V.shape[0],1)) ))
            point = np.concatenate((point,np.array([0])))
        # Distance from point to segment
        start = V[element[0],:]
        end = V[element[1],:]
        line_vec = end - start
        pnt_vec = point - start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec/line_len
        pnt_vec_scaled = pnt_vec/line_len
        t = np.dot(line_unitvec, pnt_vec_scaled)    
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = t*line_vec
        sqrD = np.sum((nearest - pnt_vec)**2.0)
        nearest = start + nearest
        lmb = [1-t,t]
    elif simplex_size==3:
        assert(dim==3)
        sqrD = 0
        tri = np.vstack((
            V[element[0],:],
            V[element[1],:],
            V[element[2],:]
        ))
        # print(tri)
        sqrD,nearest_point = pointTriangleDistance(tri,point)
        lmb = barycentric_coordinates(nearest_point,V[element[0],:],V[element[1],:],V[element[2],:])
    return sqrD,lmb
    #TODO Make it return barycentric coordinates
#    return sqrD, lmb



def pointTriangleDistance(TRI, P):
    # lifted from https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e
    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    PP0 = B + s * E0 + t * E1
    return sqrdistance, PP0