from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
# import polyscope as ps
# import igl

class TestRFTS(unittest.TestCase):
    def test_bunny_basic(self):
        mesh = "test/unit_tests_data/bunny_oded.obj"
        v, f = gpy.read_mesh(mesh)
        v = gpy.normalize_points(v)

        # Create and abstract SDF function that is the only connection to the shape
        sdf = lambda x: gpy.signed_distance(x, v, f)[0]
        n = 20
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
        # V0, E0 = gpy.marching_squares(sdf(GV), GV, n+1, n+1)
        V_mc, F_mc = gpy.marching_cubes(sdf(GV), GV, n+1, n+1, n+1)
        V0, F0 = gpy.icosphere(2)
        U,G = gpy.sdf_flow(GV, sdf, V0, F0, visualize=True)

if __name__ == '__main__':
    unittest.main()
