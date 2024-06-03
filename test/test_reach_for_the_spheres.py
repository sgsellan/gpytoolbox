from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


class TestReachForTheSpheres(unittest.TestCase):
    # def test_beat_marching_cubes_low_res(self):
    #     meshes = ["bunny_oded.obj", "spot.obj", "teddy.obj"]
    #     ns = [10, 20, 30]
    #     for mesh in meshes:
    #         for n in ns:
    #             v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
    #             v = gpy.normalize_points(v)

    #             # Create and abstract SDF function that is the only connection to the shape
    #             sdf = lambda x: gpy.signed_distance(x, v, f)[0]
    #             gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    #             GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    #             # V0, E0 = gpy.marching_squares(sdf(GV), GV, n+1, n+1)
    #             V_mc, F_mc = gpy.marching_cubes(sdf(GV), GV, n+1, n+1, n+1)
    #             h_mc = gpy.approximate_hausdorff_distance(V_mc, F_mc.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
    #             V0, F0 = gpy.icosphere(2)
    #             U,G = gpy.reach_for_the_spheres(GV, sdf, V0, F0, verbose=False, min_h = np.clip(1.5/n, 0.001, 0.1))
    #             h_ours = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
                
    #             # print(f"reach_for_the_spheres h: {h_ours}, MC h: {h_mc} for {mesh} with n={n}")
    #             self.assertTrue(h_ours < h_mc)

    # def test_noop(self):
    #     meshes = ["bunny_oded.obj", "spot.obj", "teddy.obj"]
    #     for mesh in meshes:
    #         v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
    #         v = gpy.normalize_points(v)

    #         sdf = lambda x: gpy.signed_distance(x, v, f)[0]
    #         n = 20
    #         gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    #         GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    #         U,G = gpy.reach_for_the_spheres(GV, sdf, v, f, verbose=False)

    #         h = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp=True)
    #         self.assertTrue(h < 2e-3)

    # def test_simple_is_sdf_violated(self):
    #     meshes = ["cube.obj", "hemisphere.obj", "cone.obj"]
    #     for mesh in meshes:
    #         n = 30
    #         v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
    #         v = gpy.normalize_points(v)

    #         # Create and abstract SDF function that is the only connection to the shape
    #         sdf = lambda x: gpy.signed_distance(x, v, f)[0]
    #         gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    #         GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    #         V0, F0 = gpy.icosphere(2)
    #         U,G = gpy.reach_for_the_spheres(GV, sdf, V0, F0, verbose=False, min_h = 0.5/n)
    #         sdf_rec = lambda x: gpy.signed_distance(x, U, G)[0]
    #         self.assertTrue(np.max(np.abs(sdf(GV)-sdf_rec(GV))) < 0.02)

    # def test_segfault(self):
    #     V,F = gpy.read_mesh("test/unit_tests_data/53159.stl")
    #     # is mesh normalized? print corners
    #     # print(np.min(V, axis=0))
    #     # print(np.max(V, axis=0))
    #     V = gpy.normalize_points(V)
    #     j = 32
    #     sdf = lambda x: gpy.signed_distance(x, V, F)[0]
    #     gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1))
    #     U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    #     V0, F0 = gpy.icosphere(2)
    #     Vr,Fr = gpy.reach_for_the_spheres(U, sdf, V0, F0, min_h = .01, verbose = False)
    #     # this should not segfault
    #     gpy.write_mesh("test_last_converged.obj", Vr, Fr)

    def test_singularity(self):
        V,F = gpy.read_mesh("test/unit_tests_data/horse.obj")
        # is mesh normalized? print corners
        # print(np.min(V, axis=0))
        # print(np.max(V, axis=0))
        V = gpy.normalize_points(V)
        j = 32
        sdf = lambda x: gpy.signed_distance(x, V, F)[0]
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
        V0, F0 = gpy.icosphere(2)
        # this should not crash, we should catch the singularity and output a the last converged mesh
        Vr,Fr = gpy.reach_for_the_spheres(U, sdf, V0, F0, min_h = .01, verbose = False, max_iter = 57)
        Vr,Fr = gpy.reach_for_the_spheres(U, sdf, V0, F0, min_h = .01, verbose = False, max_iter = 58)
        Vr_2,Fr_2 = gpy.reach_for_the_spheres(U, sdf, V0, F0, min_h = .01, verbose = False, max_iter = 59)
        # we hit a singularity in step 58, so the last two meshes should be the same
        self.assertTrue(np.all(Vr==Vr_2))
        self.assertTrue(np.all(Fr==Fr_2))


if __name__ == '__main__':
    unittest.main()
