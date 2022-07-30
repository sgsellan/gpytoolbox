from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

# TODO: Check that projection is happening (with distances to surface, maybe?)

class TestRemeshBotsch(unittest.TestCase):
    def test_bunny(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True)
        # igl.write_obj("output.obj",u,g)
        E,bd = gpytoolbox.edges(g,return_boundary_indices=True)
        # Boundary should be empty
        self.assertTrue(len(bd)==0)
        # Edge lengths should be "near" 0.01
        edge_lengths = np.linalg.norm(u[E[0,:],:] - u[E[1,:],:],axis=1)
        self.assertTrue(np.isclose(np.mean(edge_lengths)-0.01,0.0,atol=1e-3))

    # This example used to break the remesher     
    def test_chair_example(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/wooden-chair-remesher-bug.obj")
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),5,0.1,True)
        # There used to be a bunch of duplicate vertices
        sv,_,_ = gpytoolbox.remove_duplicate_vertices(u)
        # There shouldn't be now
        self.assertTrue(u.shape[0]==sv.shape[0])
        # Same thing without projection
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),5,0.1,False)
        # There used to be a bunch of duplicate vertices
        sv,_,_ = gpytoolbox.remove_duplicate_vertices(u)
        # There shouldn't be now
        self.assertTrue(u.shape[0]==sv.shape[0])

        


if __name__ == '__main__':
    unittest.main()