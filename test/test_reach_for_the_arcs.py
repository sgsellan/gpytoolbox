from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import time


class TestReachForTheArcs(unittest.TestCase):
    def test_beat_marching_cubes_low_res(self):
        meshes = ["R.npy", "bunny_oded.obj", "armadillo.obj"]
        ns = [10, 20]
        for mesh in meshes:
            for n in ns:
                print(f"mesh: {mesh}, res: {n}")
                if mesh[-3:]=="obj":
                    v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
                elif mesh[-3:]=="npy":
                    data = np.load("test/unit_tests_data/" + mesh,
                        allow_pickle=True)
                    v = data[()]['V']
                    f = data[()]['F']
                v = gpy.normalize_points(v)

                sdf = lambda x: gpy.signed_distance(x, v, f)[0]

                if mesh[-3:]=="obj":
                    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                    GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
                    V_mc, F_mc = gpy.marching_cubes(sdf(GV), GV, n+1, n+1, n+1)
                elif mesh[-3:]=="npy":
                    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                    GV = np.vstack((gx.flatten(), gy.flatten())).T
                    V_mc, F_mc = gpy.marching_squares(sdf(GV), GV, n+1, n+1)

                t0 = time.time()
                h_mc = gpy.approximate_hausdorff_distance(V_mc, F_mc.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
                print(f"MC Hausdorff time: {time.time()-t0}")
                t0 = time.time()
                U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                    local_search_iters=3,
                    parallel=True, verbose=False)
                print(f"RFTA time: {time.time()-t0}")
                t0 = time.time()
                h_ours = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
                print(f"RFTA Hausdorff time: {time.time()-t0}")

                #print(f"reach_for_the_arcs h: {h_ours}, MC h: {h_mc} for {mesh} with n={n}")
                self.assertTrue(h_ours < h_mc)


    def test_noop(self):
        meshes = ["R.npy", "bunny_oded.obj", "armadillo.obj"]
        for mesh in meshes:
            print(f"mesh: {mesh}")

            if mesh[-3:]=="obj":
                v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            elif mesh[-3:]=="npy":
                data = np.load("test/unit_tests_data/" + mesh,
                    allow_pickle=True)
                v = data[()]['V']
                f = data[()]['F']
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            n = 20

            if mesh[-3:]=="obj":
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            elif mesh[-3:]=="npy":
                gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten())).T

            t0 = time.time()
            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)
            print(f"RFTA time: {time.time()-t0}")

            t0 = time.time()
            h = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp=True)
            print(f"RFTA Hausdorff time: {time.time()-t0}")
            self.assertTrue(h < 0.1)


    def test_parallel_is_the_same(self):
        meshes = ["R.npy", "bunny_oded.obj", "armadillo.obj"]
        for mesh in meshes:
            print(f"mesh: {mesh}")

            if mesh[-3:]=="obj":
                v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            elif mesh[-3:]=="npy":
                data = np.load("test/unit_tests_data/" + mesh,
                    allow_pickle=True)
                v = data[()]['V']
                f = data[()]['F']
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            n = 20

            if mesh[-3:]=="obj":
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            elif mesh[-3:]=="npy":
                gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten())).T

            t0 = time.time()
            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=False, verbose=False)
            print(f"RFTA time: {time.time()-t0}")

            t0 = time.time()
            Up,Gp = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)
            print(f"RFTA par time: {time.time()-t0}")
            t0 = time.time()
            h_parallel = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), Up, Gp.astype(np.int32), use_cpp = True)
            print(f"RFTA par Hausdorff time: {time.time()-t0}")
            # print(f"parallel Hausdorff distance h: {h_parallel}")
            self.assertTrue(h_parallel < 1e-6)


    def test_simple_is_sdf_violated(self):
        meshes = ["cube.obj", "hemisphere.obj", "cone.obj"]
        for mesh in meshes:
            n = 30
            v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            v = gpy.normalize_points(v)

            sdf = lambda x: gpy.signed_distance(x, v, f)[0]
            gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
            GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            t0 = time.time()
            U,G = gpy.reach_for_the_arcs(GV, sdf(GV), fine_tune_iters=3,
                local_search_iters=3,
                parallel=True, verbose=False)
            print(f"RFTA time: {time.time()-t0}")

            sdf_rec = lambda x: gpy.signed_distance(x, U, G)[0]
            t0 = time.time()
            self.assertTrue(np.max(np.abs(sdf(GV)-sdf_rec(GV))) < 0.02)
            print(f"SDF assert time: {time.time()-t0}")


if __name__ == '__main__':
    unittest.main()
