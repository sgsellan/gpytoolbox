import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestSignedDistancePolygon(unittest.TestCase):
    # This isn't too complex, probably could use being expanded
    def test_synthetic(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])
        sample_points = np.array([  [0.0,0.0],
                                    [0.3,0.0],
                                    [-1.5,0.5],
                                    [1.2,0.0]])
        groundtruth_vals = np.array([-1.0,-0.7,0.5,0.2])
        S = gpytoolbox.signed_distance_polygon(sample_points,V)
        self.assertTrue(np.isclose(S-groundtruth_vals,0).all())
    def test_duplicated(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0],[-1.0, -1.0] ])
        sample_points = np.array([  [0.0,0.0],
                                    [0.3,0.0],
                                    [-1.5,0.5],
                                    [1.2,0.0]])
        groundtruth_vals = np.array([-1.0,-0.7,0.5,0.2])
        S = gpytoolbox.signed_distance_polygon(sample_points,V)
        self.assertTrue(np.isclose(S-groundtruth_vals,0).all())
        

if __name__ == '__main__':
    unittest.main()
