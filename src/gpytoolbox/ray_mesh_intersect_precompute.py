import numpy as np


class ray_mesh_intersect_precompute:
    """Precomputed intersector for repeated ray-mesh queries.

    Uses libigl's `igl::embree::EmbreeIntersector` when GPyToolbox was built
    with Embree. Otherwise, it builds and reuses GPyToolbox's portable AABB
    tree. Both backends avoid rebuilding their acceleration structure on
    every `ray_mesh_intersect` call.

    Parameters
    ----------
    V : (v,3) numpy double array
        Vertex positions of a 3D triangle mesh.
    F : (f,3) numpy int array
        Triangle indices into V.

    See Also
    --------
    ray_mesh_intersect, squared_distance_precompute

    Examples
    --------
    Build once, then pass to `ray_mesh_intersect` on every iteration:
    ```python
    v, f = gpytoolbox.read_mesh("bunny.obj")
    rmi = gpytoolbox.ray_mesh_intersect_precompute(v, f)
    for _ in range(num_iters):
        ts, ids, lambdas = gpytoolbox.ray_mesh_intersect(
            origins, dirs, v, f, intersector=rmi)
    ```

    The intersector can also be queried directly:
    ```python
    rmi = gpytoolbox.ray_mesh_intersect_precompute(v, f)
    ts, ids, lambdas = rmi.intersect(origins, dirs)
    ```
    """

    def __init__(self, V, F):
        V = np.ascontiguousarray(V, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.int32)
        if V.shape[1] != 3 or F.shape[1] != 3:
            raise ValueError("ray_mesh_intersect_precompute requires a 3D triangle mesh.")

        import gpytoolbox_bindings
        self._impl = None
        self._aabb = None
        if getattr(gpytoolbox_bindings, "_has_embree", True):
            from gpytoolbox_bindings import _RayMeshIntersector_cpp_impl
            self._impl = _RayMeshIntersector_cpp_impl(V, F)
        else:
            from gpytoolbox.initialize_aabbtree import initialize_aabbtree
            C, W, CH, _, _, tri_ind, _ = initialize_aabbtree(V, F=F)
            self._aabb = (C, W, CH, tri_ind)

        self._V = V
        self._F = F

    @property
    def V(self):
        return self._V

    @property
    def F(self):
        return self._F

    def intersect(self, origins, directions):
        """Intersect rays with the stored mesh.

        Parameters
        ----------
        origins : (n,3) numpy double array
            Ray origins.
        directions : (n,3) numpy double array
            Ray directions (do not need to be normalized).

        Returns
        -------
        ts : (n,) numpy double array
            Distance along each ray to the first hit (inf if no hit).
        ids : (n,) numpy int array
            Index into F of the hit triangle (-1 if no hit).
        lambdas : (n,3) numpy double array
            Barycentric coordinates of the hit point on the triangle.
        """
        origins = np.ascontiguousarray(np.atleast_2d(origins), dtype=np.float64)
        directions = np.ascontiguousarray(np.atleast_2d(directions),
                                          dtype=np.float64)
        if self._impl is not None:
            return self._impl.intersect(origins, directions)

        from gpytoolbox.ray_mesh_intersect import ray_mesh_intersect
        C, W, CH, tri_ind = self._aabb
        return ray_mesh_intersect(
            origins, directions, self._V, self._F, use_embree=False,
            C=C, W=W, CH=CH, tri_ind=tri_ind)
