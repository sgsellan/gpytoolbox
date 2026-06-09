from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


class TestReachForTheArcs(unittest.TestCase):
    def test_beat_marching_cubes_low_res(self):
        meshes = ["R.npy", "bunny_oded.obj", "armadillo.obj"]
        for mesh in meshes:
            if mesh[-3:]=="obj":
                v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            elif mesh[-3:]=="npy":
                data = np.load("test/unit_tests_data/" + mesh,
                    allow_pickle=True)
                v = data[()]['V']
                f = data[()]['F']
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            n = 10

            if mesh[-3:]=="obj":
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
                V_mc, F_mc = gpy.marching_cubes(sdf(GV), GV, n+1, n+1, n+1)
            elif mesh[-3:]=="npy":
                gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten())).T
                V_mc, F_mc = gpy.marching_squares(sdf(GV), GV, n+1, n+1)

            h_mc = gpy.approximate_hausdorff_distance(V_mc, F_mc.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)
            h_ours = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
            
            #print(f"reach_for_the_arcs h: {h_ours}, MC h: {h_mc} for {mesh} with n={n}")
            self.assertTrue(h_ours < h_mc)


    def test_noop(self):
        meshes = ["R.npy", "bunny_oded.obj", "armadillo.obj"]
        for mesh in meshes:
            if mesh[-3:]=="obj":
                v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            elif mesh[-3:]=="npy":
                data = np.load("test/unit_tests_data/" + mesh,
                    allow_pickle=True)
                v = data[()]['V']
                f = data[()]['F']
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            n = 10

            if mesh[-3:]=="obj":
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            elif mesh[-3:]=="npy":
                gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten())).T

            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)
            
            h = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp=True)
            self.assertTrue(h < 0.2)


    def test_parallel_is_the_same(self):
        meshes = ["R.npy", "bunny_oded.obj", "armadillo.obj"]
        for mesh in meshes:
            if mesh[-3:]=="obj":
                v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            elif mesh[-3:]=="npy":
                data = np.load("test/unit_tests_data/" + mesh,
                    allow_pickle=True)
                v = data[()]['V']
                f = data[()]['F']
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            n = 10

            if mesh[-3:]=="obj":
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            elif mesh[-3:]=="npy":
                gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten())).T

            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=False, verbose=False)
            
            Up,Gp = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)
            h_parallel = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), Up, Gp.astype(np.int32), use_cpp = True)
            # print(f"parallel Hausdorff distance h: {h_parallel}")
            self.assertTrue(h_parallel < 1e-6)


    def test_simple_is_sdf_violated(self):
        meshes = ["cube.obj", "hemisphere.obj"]
        for mesh in meshes:
            n = 10
            v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
            GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)

            sdf_rec = lambda x: gpy.signed_distance(x, U, G)[0]
            # print(np.max(np.abs(sdf(GV)-sdf_rec(GV))))
            self.assertTrue(np.max(np.abs(sdf(GV)-sdf_rec(GV))) < 0.05)

    def test_issue_147(self):
        # A coarse, reinitialized SDF can leave fewer than two feasible points
        # after fine tuning. reach_for_the_arcs used to crash on this input
        # (AssertionError in point_cloud_to_mesh / UnboundLocalError). It should
        # now gracefully return an empty mesh instead.
        grid_path = "test/unit_tests_data/issue-147/grid_1.txt"
        sdf_path = "test/unit_tests_data/issue-147/sdf_1.txt"
        grid = np.loadtxt(grid_path, dtype=float)
        sdf = np.loadtxt(sdf_path, dtype=float)
        d = grid.shape[1]
        vr, fr = gpy.reach_for_the_arcs(grid, sdf, verbose=True)
        # No mesh can be reconstructed from so few points: expect empty, but
        # well-formed (d-column) arrays rather than an exception.
        self.assertEqual(vr.ndim, 2)
        self.assertEqual(fr.ndim, 2)
        self.assertEqual(vr.shape[1], d)
        self.assertEqual(fr.shape[1], d)

    def test_no_feasible_points(self):
        # An SDF with no zero crossing anywhere in the domain yields no point
        # cloud at all. This used to raise UnboundLocalError; now it should
        # return an empty mesh, including when the point cloud is requested.
        gx, gy = np.meshgrid(np.linspace(0., 1., 6), np.linspace(0., 1., 6))
        U = np.stack([gx.ravel(), gy.ravel()], axis=1)
        S = np.full(U.shape[0], 5.0)  # far from any surface, all positive
        V, F = gpy.reach_for_the_arcs(U, S, verbose=True)
        self.assertEqual(V.shape, (0, 2))
        self.assertEqual(F.shape, (0, 2))
        V, F, P, N = gpy.reach_for_the_arcs(U, S, return_point_cloud=True)
        self.assertEqual(V.shape, (0, 2))
        self.assertEqual(F.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
