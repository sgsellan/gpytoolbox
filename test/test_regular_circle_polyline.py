from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestRegularCirclePolyline(unittest.TestCase):
    def test_valid_polyline(self):
        # Generate meshes of very diverse sizes
        for n in range(5,50,5):
            V,E = gpytoolbox.regular_circle_polyline(n)
            # Check: vertices are between minus one and one
            self.assertTrue(np.max(V) <= 1.0)
            self.assertTrue(np.min(V) >= -1.0)
            # Check: vertices all on circle
            self.assertTrue(np.isclose(np.linalg.norm(V, axis=-1), 1.).all())
            # Check: all edges properly oriented
            normals = np.cross( V[E[:,0],:] - 0, V[E[:,1],:] - V[E[:,0],:] , axis=1)
            self.assertTrue((normals>0).all())

if __name__ == '__main__':
    unittest.main()