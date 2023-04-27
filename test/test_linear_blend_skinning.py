from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestLinearBlendSkinning(unittest.TestCase):
    def test_two_triangles(self):
        V = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
        F = np.array([[0,1,2],[1,2,3]])
        # Create a set of handles
        Rs = np.array([np.eye(3),np.eye(3)])
        Ts = np.array([[0,0,0],[1,0,0]])
        # Create a set of weights
        Ws = np.array([[1,0],[0,1],[0,1],[1,0]])
        # Deform the mesh
        U = gpy.linear_blend_skinning(V,Ws,Rs,Ts)
        print(U)
        # Check the result
        assert np.allclose(U,np.array([[0,0,0],[2,0,0],[1,1,0],[1,1,0]]))
    # Would be nice to have a more principled test


if __name__ == '__main__':
    unittest.main()