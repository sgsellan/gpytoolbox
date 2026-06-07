import numpy as np
from gpytoolbox.edge_indices import edge_indices


class squared_distance_precompute:
    """Precomputed AABB tree for repeated closest-point queries.

    Wraps libigl's C++ `igl::AABB` so the bounding volume hierarchy is built
    once and reused across many `squared_distance` / `signed_distance` /
    `winding_number` calls (when paired with a `fast_winding_number_precompute`),
    avoiding the per-call O(n) tree construction cost.

    Parameters
    ----------
    V : (v,dim) numpy double array
        Matrix of mesh/polyline coordinates (dim must be 2 or 3).
    F : (f,s) numpy int array, optional (default None)
        Matrix of mesh/polyline indices into V. If None and V is 2D, V is
        treated as an ordered closed polyline; if None and V is 3D, V is
        treated as a point cloud.

    See Also
    --------
    squared_distance, signed_distance, fast_winding_number_precompute

    Examples
    --------
    Build once, then pass to `squared_distance` / `signed_distance` on
    every iteration:
    ```python
    v, f = gpytoolbox.read_mesh("bunny.obj")
    tree = gpytoolbox.squared_distance_precompute(v, f)
    for _ in range(num_iters):
        P = 2*np.random.rand(num_samples,3)-4
        sqrD, I, lmbs = gpytoolbox.squared_distance(P, v, F=f, use_cpp=True, aabb=tree)
    ```

    The tree can also be queried directly without going through the
    `squared_distance` wrapper:
    ```python
    tree = gpytoolbox.squared_distance_precompute(v, f)
    sqrD, I, C = tree.squared_distance(P)  # closest-point queries
    ```
    """

    def __init__(self, V, F=None):
        try:
            from gpytoolbox_bindings import _AABBTree_cpp_impl
        except ImportError:
            raise ImportError("Gpytoolbox cannot import its C++ squared_distance_precompute binding.")

        V = np.ascontiguousarray(V, dtype=np.float64)
        dim = V.shape[1]
        if F is None:
            if dim == 2:
                F = edge_indices(V.shape[0], closed=True)
            else:
                F = np.arange(V.shape[0], dtype=np.int32)[:, None]
        F = np.ascontiguousarray(F, dtype=np.int32)

        self._V = V
        self._F = F
        self._tree = _AABBTree_cpp_impl(V, F)

    @property
    def V(self):
        return self._V

    @property
    def F(self):
        return self._F

    @property
    def dim(self):
        return self._tree.dim

    def squared_distance(self, P):
        """Compute squared distances from points P to the stored mesh.

        Parameters
        ----------
        P : (p,dim) numpy double array
            Matrix of query point positions.

        Returns
        -------
        sqrD : (p,) numpy double array
            Vector of minimum squared distances.
        I : (p,) numpy int array
            Indices into F of closest elements to each query point.
        C : (p,dim) numpy double array
            Closest points on the mesh to each query point.
        """
        P = np.ascontiguousarray(np.atleast_2d(P), dtype=np.float64)
        return self._tree.squared_distance(P)
