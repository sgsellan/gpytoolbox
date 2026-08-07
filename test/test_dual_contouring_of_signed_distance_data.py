from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


def _axis_angle_rotation_matrix(axis, angle):
    # Create a rotation matrix corresponding to the rotation around a general axis by a specified angle.
    # R = dd^T + cos(theta)*(I - dd^T) + sin(theta)*skew(d)

    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    # Components of the axis vector
    x, y, z = axis

    # Construct the skew-symmetric matrix
    skew_sym = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # Identity matrix
    I = np.eye(3)

    # Outer product of the axis vector with itself
    outer = np.outer(axis, axis)

    # Rotation matrix
    R = outer + np.cos(angle) * (I - outer) + np.sin(angle) * skew_sym

    return R


def _quads_to_tris(F):
    # Triangulate quads splitting along the diagonal from vertex 0 to vertex 2
    F = np.asarray(F)
    if F.shape[0] == 0:
        return np.zeros((0, 3), dtype=int)
    return np.vstack((F[:, [0, 1, 2]], F[:, [0, 2, 3]]))


def _box_sdf(P, half_extents):
    q = np.abs(P) - half_extents[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _meshgrid_grid(nx, ny, nz, lo=-1.0, hi=1.0):
    # dual_contouring_of_signed_distance_data needs GV ordered so the first
    # axis varies fastest, then the second, then the third.
    x = np.linspace(lo, hi, nx)
    y = np.linspace(lo, hi, ny)
    z = np.linspace(lo, hi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.stack((X.ravel(order='F'), Y.ravel(order='F'),
        Z.ravel(order='F')), axis=-1)


def _regular_cube_mesh_grid(nx, ny, nz):
    # Call regular_cube_mesh and reorder the output so the first axis varies fastest, then the second, then the third.
    GV, _ = gpy.regular_cube_mesh(nx, ny, nz)
    return np.stack([
        GV[:, d].reshape((ny, nx, nz)).transpose(1, 0, 2).ravel(order='F')
        for d in range(3)
    ], axis=-1)


class TestDualContouringOfSignedDistanceData(unittest.TestCase):
    def _test_mesh(self, mesh, n):
        V, F = gpy.read_mesh("test/unit_tests_data/" + mesh)
        V = gpy.normalize_points(V, center=np.array([0.5, 0.5, 0.5]))
        # Shrink to [0.05, 0.95]^3 to avoid clipping issues with the grid's bounding box.
        V = 0.5 + 0.9 * (V - 0.5) 

        GV = _regular_cube_mesh_grid(n, n, n)
        S = gpy.signed_distance(GV, V, F)[0]

        U, G = gpy.dual_contouring_of_signed_distance_data(
            S, GV, n, n, n
        )
        Gt = _quads_to_tris(G)

        self.assertTrue(U.shape[0] > 0)
        self.assertTrue(Gt.shape[0] > 0)

        # Save the objs
        # gpy.write_mesh("test/test_dcsdd_results/" + mesh[:-4] + f"_{n}.obj", 
        #                U, Gt)

        sdf_tol = 1.0 / (n - 1)   # Grid cell size   
        dist_tol = sdf_tol ** 2

        # print(f"Testing mesh {mesh} at resolution {n}:")
        # print(f"  SDF tolerance: {sdf_tol}")
        # print(f"  Squared distance tolerance: {dist_tol}")

        # The reconstruction's own SDF should roughly agree with the
        # original SDF at the grid vertices used to build it.
        s2 = gpy.signed_distance(GV, U, Gt)[0]
        self.assertTrue(np.isclose(S, s2, atol=sdf_tol).all())

        # And the reconstructed vertices should lie close to the input
        dists = gpy.squared_distance(U, V, F=F, use_cpp=True)[0]
        self.assertTrue(np.isclose(dists, 0.0, atol=dist_tol).all())

    def test_meshes(self):
        meshes = ["cube.obj", "bunny_oded.obj"]
        resolutions = [15, 25]
        for mesh in meshes:
            for res in resolutions:
                self._test_mesh(mesh, res)


    def _test_beat_marching_cubes_on_mesh(self, mesh, n):
        V, F = gpy.read_mesh("test/unit_tests_data/" + mesh)
        V = gpy.normalize_points(V, center=np.array([0.5, 0.5, 0.5]))
        # Shrink to [0.05, 0.95]^3 to avoid clipping issues with the grid's bounding box.
        V = 0.5 + 0.9 * (V - 0.5) 

        GV = _regular_cube_mesh_grid(n + 1, n + 1, n + 1)
        S = gpy.signed_distance(GV, V, F)[0]

        V_mc, F_mc = gpy.marching_cubes(S, GV, n + 1, n + 1, n + 1)
        h_mc = gpy.approximate_hausdorff_distance(V_mc, F_mc.astype(np.int32),
            V, F.astype(np.int32), use_cpp=True)

        U, G = gpy.dual_contouring_of_signed_distance_data(
            S, GV, n + 1, n + 1, n + 1
        )
        Gt = _quads_to_tris(G).astype(np.int32)
        h_ours = gpy.approximate_hausdorff_distance(U, Gt, V,
            F.astype(np.int32), use_cpp=True)
        
        # Save the objs for inspection
        # gpy.write_mesh("test/test_dcsdd_results/" + mesh[:-4] + f"_{n}_mc.obj",
        #                 V_mc, F_mc) 
        # gpy.write_mesh("test/test_dcsdd_results/" + mesh[:-4] + f"_{n}_ours.obj",
        #                 U, Gt)

        self.assertTrue(h_ours < h_mc)


    def test_beat_marching_cubes(self):
        # DCSDD should produce a better reconstruction than MC in cases with
        # sharp features and not too high resolutions.
        meshes = ["cube.obj", "cone.obj"]
        resolutions = [8, 12, 16]
        for mesh in meshes:
            for n in resolutions:
                # print(f"Testing that DCSDD beats MC on {mesh} at resolution {n}")
                self._test_beat_marching_cubes_on_mesh(mesh, n)


    def test_random_rotation(self):
        # Test that the accuracy of the outputs is still as expected even
        # when the input mesh is randomly rotated.
        rng = np.random.default_rng(0)
        V, F = gpy.read_mesh("test/unit_tests_data/cube.obj")
        V0 = V.copy()
        resolutions = rng.integers(8, 40, size=6)

        for n in resolutions:
            n = int(n)
            axis = rng.random(3)
            axis = axis / np.linalg.norm(axis)
            angle = float(rng.random()) * 2 * np.pi
            R = _axis_angle_rotation_matrix(axis, angle)
            V = V0 @ R
            V = gpy.normalize_points(V, center=np.array([0.5, 0.5, 0.5]))
            V = 0.5 + 0.9 * (V - 0.5)

            GV = _regular_cube_mesh_grid(n, n, n)
            S = gpy.signed_distance(GV, V, F)[0]
            U, G = gpy.dual_contouring_of_signed_distance_data(
                S, GV, n, n, n
            )
            Gt = _quads_to_tris(G).astype(np.int32)

            # Check accuracy
            h = gpy.approximate_hausdorff_distance(U, Gt, V, F.astype(np.int32),
                                                   use_cpp=True)
            # print(f"Random rotation test on cube at resolution {n}: Hausdorff distance = {h}")
            self.assertTrue(h < 0.1)


    def test_regular_cube_mesh_matches_meshgrid(self):
        # The two GV-construction recipes from the docstring's Examples
        # section -- plain np.meshgrid, and regular_cube_mesh with its
        # columns reordered -- describe the same grid and must therefore
        # produce identical reconstructions.
        nx, ny, nz = 10, 10, 10

        GV_meshgrid = _meshgrid_grid(nx, ny, nz, lo=0.0, hi=1.0)
        GV_cube_mesh = _regular_cube_mesh_grid(nx, ny, nz)
        np.testing.assert_allclose(GV_meshgrid, GV_cube_mesh)

        S = np.linalg.norm(GV_meshgrid - 0.5, axis=1) - 0.3

        V1, F1 = gpy.dual_contouring_of_signed_distance_data(
            S, GV_meshgrid, nx, ny, nz, outer_iters=3, inner_iters=3)
        V2, F2 = gpy.dual_contouring_of_signed_distance_data(
            S, GV_cube_mesh, nx, ny, nz, outer_iters=3, inner_iters=3)

        np.testing.assert_allclose(V1, V2)
        np.testing.assert_array_equal(F1, F2)

    def test_non_cubic_grid_works(self):
        nx, ny, nz = 12, 16, 20
        half_extents = np.array([0.3, 0.5, 0.7])

        GV = _meshgrid_grid(nx, ny, nz)
        S = _box_sdf(GV, half_extents)

        V, F = gpy.dual_contouring_of_signed_distance_data(
            S, GV, nx, ny, nz, outer_iters=4, inner_iters=4)

        self.assertTrue(V.shape[0] > 0)
        max_abs = np.max(np.abs(V), axis=0)
        # Tight enough to catch an axis mix-up (the smallest gap between any
        # two of half_extents' components is 0.2), loose enough for the
        # resolution used here.
        self.assertTrue(np.all(np.abs(max_abs - half_extents) < 0.15))

    def test_isovalue(self):
        n = 16
        GV = _meshgrid_grid(n, n, n)
        S = np.linalg.norm(GV, axis=1)

        for target_r in [0.3, 0.6]:
            V, F = gpy.dual_contouring_of_signed_distance_data(
                S, GV, n, n, n, isovalue=target_r, outer_iters=4,
                inner_iters=4, verbose=False)
            self.assertTrue(V.shape[0] > 0)
            radii = np.linalg.norm(V, axis=1)
            self.assertTrue(np.isclose(radii, target_r, atol=0.15).all())

    def test_invalid_inputs(self):
        nx = ny = nz = 4
        n_samples = nx * ny * nz
        GV = np.random.default_rng(0).random((n_samples, 3))
        S = np.random.default_rng(1).random(n_samples)

        # GV with the wrong number of columns.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(
                S, GV[:, :2], nx, ny, nz)

        # S and GV disagree on the number of samples.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(
                S[:-1], GV, nx, ny, nz)

        # S and GV agree with each other, but not with nx*ny*nz.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(
                S[:-1], GV[:-1, :], nx, ny, nz)

        # nx (and ny, nz) must be at least 2.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(S, GV, 1, ny, nz)

        # outer_iters must be nonnegative.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(
                S, GV, nx, ny, nz, outer_iters=-1)

        # inner_iters must be nonnegative.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(
                S, GV, nx, ny, nz, inner_iters=-1)

        # batch_size must be positive.
        with self.assertRaises(ValueError):
            gpy.dual_contouring_of_signed_distance_data(
                S, GV, nx, ny, nz, batch_size=0)


if __name__ == '__main__':
    unittest.main()
