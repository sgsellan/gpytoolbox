from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestQuadtreeBoundary(unittest.TestCase):
    def test_synthetic(self):
        np.random.seed(0)
        th = 2*np.pi*np.random.rand(500,1)
        P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)

        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
        V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)

        # Generate random query points in each cell
        coords = np.random.rand(C.shape[0],2)
        for i in range(C.shape[0]):
            rand_point = C[i,:] + 0.5*W[i]*coords[i,:]
            ind, all_ind = gpytoolbox.in_quadtree(rand_point,C,W,CH)
            if CH[i,0]==-1:
                # Then, the first index should be this one
                self.assertTrue(ind==i)
                # *and* it should be one of the indeces in the list
                self.assertTrue(i in all_ind)
            else:
                # Then this is not a child, so it should only have the index in the list of all containing indeces
                self.assertTrue(i in all_ind)


if __name__ == '__main__':
    unittest.main()
