from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestBadQuadMeshFromQuadtree(unittest.TestCase):
    def test_2d_quadtree_mesh(self):
        np.random.seed(0)
        for i in range(5):
            P = 2*np.random.rand(100,2) - 1
            C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=2,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
            child_indeces = gpytoolbox.quadtree_children(CH)
            V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
            # Check that the children in V are the children in child_indeces
            children_verts_1 = C[child_indeces,:]
            for j in range(Q.shape[0]):
                child_vert = (V[Q[j,0],:] + V[Q[j,1],:] + V[Q[j,2],:] + V[Q[j,3],:])/4
                self.assertTrue(child_vert in children_verts_1)

    def test_3d_octtree_mesh(self):
        np.random.seed(0)
        for i in range(5):
            P = 2*np.random.rand(20,3) - 1
            C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=2,vmin=np.array([-1,-1,-1]),vmax=np.array([1,1,1]))
            child_indeces = gpytoolbox.quadtree_children(CH)
            V,Q,H = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
            # Check that the children in V are the children in child_indeces
            children_verts_1 = C[child_indeces,:]
            for j in range(H.shape[0]):
                child_vert = (V[H[j,0],:] + V[H[j,1],:] + V[H[j,2],:] + V[H[j,3],:] + V[H[j,4],:] + V[H[j,5],:] + V[H[j,6],:] + V[H[j,7],:])/8
                self.assertTrue(child_vert in children_verts_1)
            
if __name__ == '__main__':
    unittest.main()
