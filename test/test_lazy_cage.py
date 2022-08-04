from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestLazyCage(unittest.TestCase):
    def test_bunny(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # Generate many examples
        for m in np.linspace(20,2000,10,dtype=int):
            # print(m)
            u,g = gpytoolbox.copyleft.lazy_cage(v,f,num_faces=m,max_iter=10)
            # u,g fully contains v,f, i.e. they don't intersect
            b, _ = gpytoolbox.copyleft.do_meshes_intersect(v,f,u,g)
            self.assertFalse(b)
    def test_armadillo(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/armadillo.obj")
        # print(v)
        # Generate many examples
        for m in np.linspace(1000,2000,5,dtype=int):
            # print(m)
            u,g = gpytoolbox.copyleft.lazy_cage(v,f,num_faces=m,max_iter=10)
            # u,g fully contains v,f, i.e. they don't intersect
            b, _ = gpytoolbox.copyleft.do_meshes_intersect(v,f,u,g)
            self.assertFalse(b)
            
if __name__ == '__main__':
    unittest.main()