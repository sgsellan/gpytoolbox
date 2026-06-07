import numpy as np


class ray_mesh_intersect_precompute:
    """Precomputed Embree intersector for repeated ray-mesh queries.

    Wraps libigl's `igl::embree::EmbreeIntersector` so the Embree scene is
    built once and reused across many `ray_mesh_intersect` calls, avoiding
    the per-call construction cost.

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
        try:
            from gpytoolbox_bindings import _RayMeshIntersector_cpp_impl
        except ImportError:
            raise ImportError(
                "Gpytoolbox cannot import its C++ ray_mesh_intersect_precompute binding.")

        V = np.ascontiguousarray(V, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.int32)
        if V.shape[1] != 3 or F.shape[1] != 3:
            raise ValueError("ray_mesh_intersect_precompute requires a 3D triangle mesh.")

        self._V = V
        self._F = F
        self._impl = _RayMeshIntersector_cpp_impl(V, F)

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
        return self._impl.intersect(origins, directions)
