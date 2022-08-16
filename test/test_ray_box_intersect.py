from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

# Would be nice to make this test more detailed
class TestRayBoxIntersect(unittest.TestCase):
    def test_simple_box(self):
        th = 2*np.pi*np.random.rand(1000) - np.pi
        center = np.array([0,0])
        width = np.array([1,1])
        position = np.array([-1,0])
        for i in range(th.shape[0]):
            dir = np.array([np.cos(th[i]),np.sin(th[i])])
            is_hit,where_hit = gpytoolbox.ray_box_intersect(position,dir,center,width)
            if np.abs(th[i])<=(np.pi/4):
                self.assertTrue(is_hit)
            else:
                self.assertFalse(is_hit)
    def test_inside(self):
        P = np.random.rand(100,2) - 0.5
        th = 2*np.pi*np.random.rand(100) - np.pi
        center = np.array([0,0])
        width = np.array([1,1])
        for i in range(P.shape[0]):
            dir = np.array([np.cos(th[i]),np.sin(th[i])])
            is_hit,where_hit = gpytoolbox.ray_box_intersect(P[i,:],dir,center,width)
            self.assertTrue(is_hit)
            
            
        
if __name__ == '__main__':
    unittest.main()