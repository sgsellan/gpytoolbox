from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestVolume(unittest.TestCase):
    def test_single_tet(self):
        v = np.array([[0.0,0.0,0.0],
                      [1.0,0.0,0.0],
                      [0.0,1.0,0.0],
                      [0.0,0.0,1.0]
                      ])
        # This tet is properly oriented
        t = np.array([[0,1,2,3]])
        vols = gpytoolbox.volume(v,t)
        self.assertTrue(vols==np.array([1/6]))
        # This tet is not properly oriented
        t_backwards = np.array([[0,1,3,2]])
        vols = gpytoolbox.volume(v,t_backwards)
        self.assertTrue(vols==np.array([-1/6]))

    def test_regular_cube_mesh(self):
        v,t = gpytoolbox.regular_cube_mesh(11)
        vols = gpytoolbox.volume(v,t)
        # Check against analytical volume
        self.assertTrue(np.isclose(vols - (0.1**3.)/6.,0.0).all())


if __name__ == '__main__':
    unittest.main()