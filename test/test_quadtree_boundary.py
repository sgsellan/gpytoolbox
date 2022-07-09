from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestQuadtreeBoundary(unittest.TestCase):
    def test_is_boundary(self):
        np.random.seed(0)
        th = 2*np.pi*np.random.rand(100,1)
        P = 2*np.random.rand(100,2) - 1

        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=2,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
        V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
        bd_children, bd_all = gpytoolbox.quadtree_boundary(CH,A)

        # Are all the indeces boundary?
        for i in range(len(bd_all)):
            all_corner_values = np.array([C[bd_all[i],0]+.5*W[bd_all[i]],
                                        C[bd_all[i],0]-.5*W[bd_all[i]],
                                        C[bd_all[i],1]+.5*W[bd_all[i]],
                                        C[bd_all[i],1]-.5*W[bd_all[i]]])
            # corner furthest from the center
            furthest_corner = np.amax(np.abs(all_corner_values))
            self.assertTrue(np.isclose(furthest_corner,1.0))

    def test_is_child(self):
        np.random.seed(0)
        th = 2*np.pi*np.random.rand(100,1)
        P = 2*np.random.rand(100,2) - 1

        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=2,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
        V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
        bd_children, bd_all = gpytoolbox.quadtree_boundary(CH,A)
        # Are all the children indeces children?
        for i in range(len(bd_children)):
            self.assertTrue(CH[bd_children[i],0]==-1)


if __name__ == '__main__':
    unittest.main()
