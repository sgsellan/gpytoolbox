import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestHausdorffDistance(unittest.TestCase):
    def test_bunny(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        v = gpytoolbox.normalize_points(v)
        v = 0.7*v
        GV,GF = gpytoolbox.regular_cube_mesh(100)
        GV = gpytoolbox.normalize_points(GV)
        S = gpytoolbox.signed_distance(GV,v,f,use_cpp=True)
        u,g = gpytoolbox.marching_cubes(S,0.0)
    

if __name__ == '__main__':
    unittest.main()
