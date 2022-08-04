from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import scipy.sparse
import scipy.sparse.csgraph

class TestMeshBoolean(unittest.TestCase):
    # Would be nice to make this better with more tests, especially groundtruths. Let's improve it once we have more functions (e.g., signed distances)
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
            u,g = gpytoolbox.copyleft.mesh_boolean(v1,f1,v2+np.tile(displacement,(v2.shape[0],1)),f2,boolean_type='union')
            TT, TTi = gpytoolbox.triangle_triangle_adjacency(g)
            I = np.squeeze(np.reshape(TT,(-1,1),order='F'))
            J = np.linspace(0,g.shape[0]-1,g.shape[0],dtype=int)
            J = np.concatenate((J,J,J))
            vals = np.ones(J.shape)
            # Adjacency matrix
            A = scipy.sparse.csr_matrix((vals,(I,J)))
            n, labs = scipy.sparse.csgraph.connected_components(A)
            # If the displacement is <=1, then there is intersection. Otherwise, no
            uu,gg = gpytoolbox.copyleft.mesh_boolean(v1,f1,v2+np.tile(displacement,(v2.shape[0],1)),f2,boolean_type='intersection')
            if np.max(np.abs(displacement))<=1:
                self.assertTrue(n==1)
                self.assertTrue(len(uu)>0)
            else:
                self.assertTrue(len(uu)==0)
                self.assertTrue(n==2)
        
if __name__ == '__main__':
    unittest.main()