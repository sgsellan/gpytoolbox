from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

# Would be nice to make this test more detailed
class TestRayTriangleIntersect(unittest.TestCase):
    def test_against_embree(self):
        # Generate random query and triangle
        for i in range(10000):
            origin = np.random.rand(3)
            dir = np.random.rand(3)
            V = np.random.rand(3,3)
            v1 = V[0,:]
            v2 = V[1,:]
            v3 = V[2,:]
            t,is_hit,hit_coord = gpytoolbox.ray_triangle_intersect(origin,dir,v1,v2,v3)
            F = np.array([[0,1,2]])
            t2,_,lmb = gpytoolbox.ray_mesh_intersect(origin[None,:],dir[None,:],V,F)
            if t2<np.Inf:
                self.assertTrue(np.isclose(t-t2,0,atol=1e-4))
                self.assertTrue(np.isclose(origin + t*dir - hit_coord,0,atol=1e-4).all())
                self.assertTrue(is_hit)
            else:
                self.assertTrue(t==np.Inf)
                self.assertFalse(is_hit)

        
            
            
        
if __name__ == '__main__':
    unittest.main()