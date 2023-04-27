import numpy as np
from .context import gpytoolbox
from .context import unittest
import matplotlib.pyplot as plt

class TestWindingNumber(unittest.TestCase):
    def test_simple_pngs(self):
        # Build a polyline
        filename = "test/unit_tests_data/illustrator.png"
        poly = gpytoolbox.png2poly(filename)[0]
        poly = np.vstack((poly,poly + np.array([50,0])))
        # normalize
        poly = gpytoolbox.normalize_points(poly)
        
        # Make a grid between -1 and 1
        x = np.linspace(-1,1,100)
        y = np.linspace(-1,1,100)
        X,Y = np.meshgrid(x,y)
        # Vertices in the grid
        V = np.vstack((X.flatten(),Y.flatten())).T
        # Compute winding number
        W = gpytoolbox.winding_number(poly,gpytoolbox.edge_indices(poly.shape[0],closed=True),V)
        # Reshape
        W = W.reshape(X.shape)
        # Plot
        self.assertTrue(False)

        

if __name__ == '__main__':
    unittest.main()
