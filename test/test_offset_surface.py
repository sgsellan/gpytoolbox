from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestOffsetSurface(unittest.TestCase):
    # This is not a great test. We should improve it when we can measure distances and stuff.
    def test_bunny(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # Generate many examples
        for m in np.linspace(20,2000,10,dtype=int):
            # print(m)
            u1,g1 = gpytoolbox.offset_surface(v,f,iso=0.05,grid_size=50)
            u2,g2 = gpytoolbox.offset_surface(v,f,iso=0.1,grid_size=50)
            # u1,g1 fully contains u2,g2, i.e. they don't intersect
            b, _ = gpytoolbox.copyleft.do_meshes_intersect(u1,g1,u2,g2)
            self.assertFalse(b)
    def test_armadillo(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/armadillo.obj")
        v = gpytoolbox.normalize_points(v)
        # print(v)
        # Generate many examples
        sizes = [50, 100, 150]
        for gs in sizes:
            for m in np.linspace(1000,2000,5,dtype=int):
                # print(m)
                u1,g1 = gpytoolbox.offset_surface(v,f,iso=0.05,grid_size=gs)
                u2,g2 = gpytoolbox.offset_surface(v,f,iso=0.1,grid_size=gs)
                # u1,g1 fully contains v,f, i.e. they don't intersect
                b, _ = gpytoolbox.copyleft.do_meshes_intersect(u1,g1,u2,g2)
                self.assertFalse(b)
            
if __name__ == '__main__':
    unittest.main()