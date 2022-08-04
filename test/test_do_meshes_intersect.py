from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestDoMeshesIntersect(unittest.TestCase):
    def test_cubes(self):
        np.random.seed(0)
        # Build two one by one cubes
        v1,f1 = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        v1 = gpytoolbox.normalize_points(v1,center=np.array([0.5,0.5,0.5]))
        v2,f2 = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        v2 = gpytoolbox.normalize_points(v1,center=np.array([0.5,0.5,0.5]))
        for i in range(100):
            # Generate random displacements
            displacement = 4*np.random.rand(1,3)-2
            b,_ = gpytoolbox.copyleft.do_meshes_intersect(v1,f1,v2+np.tile(displacement,(v2.shape[0],1)),f2)
            # If the displacement is <=1, then there is intersection. Otherwise, no
            if np.max(np.abs(displacement))<=1:
                self.assertTrue(b)
            else:
                self.assertFalse(b)
        # If one cube is fully contained in the other, it should not return an intersection
        b,_ = gpytoolbox.copyleft.do_meshes_intersect(v1,f1,0.98*v2 + 0.01,f2)
        self.assertFalse(b)
        # Even if the cubes share a single point, the exact predicates should catch it
        displacement = np.array([1.0,0.25,0.25])
        b,_ = gpytoolbox.copyleft.do_meshes_intersect(v1,f1,v2+np.tile(displacement,(v2.shape[0],1)),f2)
        self.assertTrue(b)
        
if __name__ == '__main__':
    unittest.main()