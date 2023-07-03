from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
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
        # Check the result
        self.assertTrue(np.allclose(U,np.array([[0,0,0],[2,0,0],[1,1,0],[1,1,0]])))
    
    def test_rigid_transforms(self):
        rng = np.random.default_rng(653)
        for mesh in ['airplane', 'armadillo', 'bunny_oded', 'cube', 'mountain']:
            V,F = gpy.read_mesh("test/unit_tests_data/"+mesh+".obj")
            n = V.shape[0]
            k = rng.integers(4,20)
            Ws = rng.uniform(size=(n,k))
            Ws /= np.sum(Ws, axis=-1)[:,None]
            trans = np.tile(rng.uniform(size=3)-0.5, (k,1))
            rot  = np.tile(sp.spatial.transform.Rotation.random(None,rng).as_matrix(), (k,1,1))
            U = gpy.linear_blend_skinning(V,Ws,rot,trans)

            Vmap = V@rot[0,:].T + trans[0,:][None,...]

            self.assertTrue(np.allclose(U,Vmap))

if __name__ == '__main__':
    unittest.main()