import numpy as np


class FastWindingNumberBVH:
    """Precomputed BVH for repeated fast winding number queries.

    Wraps libigl's C++ `igl::FastWindingNumberBVH` so the bounding volume
    hierarchy is built once and reused across many `fast_winding_number` /
    `winding_number` / `signed_distance` calls, avoiding the per-call O(n)
    precomputation cost (Barill et al. "Fast Winding Numbers for Soups and
    Clouds", SIGGRAPH 2018).

    Parameters
    ----------
    V : (v,3) numpy double array
        Matrix of mesh vertex coordinates (must be 3D).
    F : (f,3) numpy int array
        Matrix of triangle indices into V.
    order : int, optional (default 2)
        Taylor series expansion order used for the BVH (supports 0, 1, 2).

    See Also
    --------
    fast_winding_number, winding_number, signed_distance, AABBTree

    Examples
    --------
    Build once, then pass to `fast_winding_number` / `winding_number` /
    `signed_distance` on every iteration:
    ```python
    v, f = gpytoolbox.read_mesh("bunny.obj")
    bvh = gpytoolbox.FastWindingNumberBVH(v, f)
    for _ in range(num_iters):
        Q = 2*np.random.rand(num_samples,3)-4
        W = gpytoolbox.fast_winding_number(Q, v, f, fwn_bvh=bvh)
    ```

    The BVH can also be queried directly without going through the
    `fast_winding_number` wrapper:
    ```python
    bvh = gpytoolbox.FastWindingNumberBVH(v, f)
    W = bvh.winding_number(Q)
    ```
    """

    def __init__(self, V, F, order=2):
        try:
            from gpytoolbox_bindings import _FastWindingNumberBVH_cpp_impl
        except ImportError:
            raise ImportError(
                "Gpytoolbox cannot import its C++ FastWindingNumberBVH binding.")

        V = np.ascontiguousarray(V, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.int32)
        if V.shape[1] != 3 or F.shape[1] != 3:
            raise ValueError(
                "FastWindingNumberBVH only supports 3D triangle meshes (V must "
                "have 3 columns and F must have 3 columns).")

        self._V = V
        self._F = F
        self._order = int(order)
        self._bvh = _FastWindingNumberBVH_cpp_impl(V, F, self._order)

    @property
    def V(self):
        return self._V

    @property
    def F(self):
        return self._F

    @property
    def order(self):
        return self._order

    def winding_number(self, Q, accuracy_scale=2.0):
        """Compute the fast winding number at points Q.

        Parameters
        ----------
        Q : (q,3) numpy double array
            Matrix of query point positions.
        accuracy_scale : float, optional (default 2.0)
            Barnes-Hut style parameter separating near and far field. Higher
            values give more accurate but slower evaluation.

        Returns
        -------
        W : (q,) numpy double array
            Vector of winding numbers (~0 outside, ~1 inside).
        """
        Q = np.ascontiguousarray(np.atleast_2d(Q), dtype=np.float64)
        return self._bvh.winding_number(Q, float(accuracy_scale))
