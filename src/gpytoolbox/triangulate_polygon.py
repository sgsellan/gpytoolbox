import numpy as np

def triangulate_polygon(V,
                        F=None,
                        a=0.,
                        q=np.pi/8.,
                        steiner=True):
    """Triangulates a two-dimensional polygon

    Computes a constrained Delaunay triangulation of the region bounded by a
    given polygon, optionally refined to satisfy a maximum area and a minimum
    angle constraint.

    Parameters
    ----------
    V : numpy double array
        Matrix of polygon vertex coordinates (must be two-dimensional)
    F : numpy int array, optional (default None)
        Matrix of polygon edge indices into V. If None, the convex hull of V
        is triangulated. Holes are detected automatically: a closed loop
        nested inside another one is a hole, a closed loop nested inside a
        hole is filled again, and so on
    a : double, optional (default 0.0)
        Maximum area of any output triangle. If 0., there is no maximum area
        constraint
    q : double, optional (default np.pi/8)
        Minimum angle of any output triangle, in radians
    steiner : bool, optional (default True)
        Whether the insertion of Steiner points on the polygon boundary is
        allowed

    Returns
    -------
    V2 : numpy double array
        Matrix of triangle mesh vertex coordinates (two-dimensional)
    F2 : numpy int array
        Matrix of triangle mesh indices into V2

    Notes
    -----
    This uses Artem Amirkhanov's (artem-ogre) CDT library,
    [https://github.com/artem-ogre/CDT](https://github.com/artem-ogre/CDT).
    The syntax is inspired by libigl's `triangulate.h`,
    [https://github.com/libigl/libigl/blob/main/include/igl/triangle/triangulate.h](https://github.com/libigl/libigl/blob/main/include/igl/triangle/triangulate.h).
    CDT does not refine, so the area and angle constraints are enforced by
    Ruppert's algorithm on top of it, following J. Ruppert, "A Delaunay
    Refinement Algorithm for Quality 2-Dimensional Mesh Generation", Journal
    of Algorithms 18(3), 1995.

    The vertices in V must be distinct, and the edges in F must not cross each
    other except at their endpoints, so a closed polyline whose last
    vertex repeats its first must have that repeat removed.

    Every vertex of V ends up in the output, used by its triangles, including
    the vertices that no edge of F refers to. Appending such points to V is how
    you force the triangulation to contain points of your choosing. They have to
    lie inside the polygon: a vertex outside it, or inside one of its holes,
    raises a `ValueError`. A vertex within a rounding error of a hole's
    boundary counts as lying on the boundary, and is included.

    Refinement to ensure a and q limits hold stops after 100 passes or 1000000
    vertices, whichever comes first.

    Examples
    --------
    ```python
    # Triangulate the unit square
    V = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    F = np.array([[0,1],[1,2],[2,3],[3,0]])
    V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.01)
    ```
    """

    try:
        from gpytoolbox_bindings import _triangulate_polygon_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    V = np.asarray(V,dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("V must be a matrix of two-dimensional vertex "
            "coordinates.")

    if F is None:
        F = np.zeros((0,2),dtype=np.int32)
    F = np.asarray(F,dtype=np.int32)
    if F.ndim != 2 or F.shape[1] != 2:
        raise ValueError("F must be a matrix of edge indices into V.")

    if a < 0.0:
        raise ValueError("a must be nonnegative.")

    if q < 0.0 or q >= np.pi/3.:
        raise ValueError("q must be between zero and pi/3.")

    V2, F2 = _triangulate_polygon_cpp_impl(V,F,float(a),float(q),bool(steiner))

    return V2, F2
