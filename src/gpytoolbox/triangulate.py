import numpy as np

def triangulate(V,E=None,
    max_area=None,
    min_angle=None,
    max_steiner_points=None):
    """Create a conforming & constrained Delaunay triangulation of the input
    curve in 2D.

    This is a wrapper for the [CDT library](https://github.com/MarcoAttene/CDT).

    Parameters
    ----------
    V : (n,2) numpy array
        vertex list of points to mesh
    E : (m,2) numpy int array, optional (default None)
        edge index list of a polyline.
        If this is None, will simply mesh the convex hull of V.
    max_area: float, optional (default None)
        If this is not None, the method will refine until this is the maximal
        area of any triangle. 
        NOTE: This curently does nothing.
    min_angle : float, optional (default None)
        If this is not None, the method will refine until this is the minimal
        angle (in radians) of any triangle. 
        If this value is too aggressive, the method might silently fail to respect
        the threshold.
        NOTE: This curently does nothing.
    max_steiner_points : int, optional (default (6*n)**2)
        how many points can be added during the refinement process to fulfill
        max_area and min_angle
        NOTE: This curently does nothing.

    Returns
    -------
    W : numpy double array
        Matrix of mesh vertices
    F : numpy int array
        Matrix of triangle indices

    Notes
    -----
    max_area, min_angle, and max_steiner_points are not yet supported.
    Once CDT adds support for them, add it to this function.


    Examples
    --------
    ```python
    import gpytoolbox as gpy
    # Polyline in V,E
    W,F = gpy.triangulate(V,E)
    # New mesh in W,F
    ```
    """

    # Try to import C++ binding
    try:
        from gpytoolbox_bindings import _triangulate_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ triangulate binding.")

    assert len(V.shape)==2 and V.shape[1]==2, "V must be a list of points in 2D."
    assert max_area is None or max_area>0, "max_area must be either None or a positive number."
    if max_area is None:
        max_area = -1.
    assert min_angle is None or min_angle>=0, "min_angle must be either None or a nonnegative number."
    if min_angle is None:
        min_angle = -1.
    if max_steiner_points is None:
        max_steiner_points = (6*V.shape[0])**2
    assert max_steiner_points>=0

    if E is None:
        E = np.array([], dtype=int)

    W,F = _triangulate_cpp_impl(V.astype(np.float64),E.astype(np.int32),
        max_area, min_angle, max_steiner_points)

    return W,F
