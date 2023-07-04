from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
# import polyscope as ps
# import igl

class TestRFTS(unittest.TestCase):
    def test_beat_marching_cubes_low_res(self):
        meshes = ["bunny_oded.obj", "spot.obj", "teddy.obj"]
        ns = [10, 20, 30, 40]
        for mesh in meshes:
            for n in ns:
        # mesh = "test/unit_tests_data/bunny_oded.obj"
                v, f = gpy.read_mesh("test/unit_tests_data/" + mesh)
                v = gpy.normalize_points(v)

                # Create and abstract SDF function that is the only connection to the shape
                sdf = lambda x: gpy.signed_distance(x, v, f)[0]
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
                # V0, E0 = gpy.marching_squares(sdf(GV), GV, n+1, n+1)
                V_mc, F_mc = gpy.marching_cubes(sdf(GV), GV, n+1, n+1, n+1)
                h_mc = gpy.approximate_hausdorff_distance(V_mc, F_mc.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
                V0, F0 = gpy.icosphere(2)
                U,G = gpy.sdf_flow(GV, sdf, V0, F0, verbose=False, min_h = 2.0/n)
                h_ours = gpy.approximate_hausdorff_distance(U, G.astype(np.int32), v, f.astype(np.int32), use_cpp = True)
                # print("Hausdorff distance between mesh and marching cubes: ", h_mc)
                # print("Hausdorff distance between mesh and our method: ", h_ours)
                self.assertTrue(h_ours < h_mc)

if __name__ == '__main__':
    unittest.main()
