from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestQuadtreeChildren(unittest.TestCase):
    def test_are_children(self):
        np.random.seed(0)
        P = 2*np.random.rand(100,2) - 1
        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=2,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
        child_indeces = gpytoolbox.quadtree_children(CH)
        for i in range(len(child_indeces)):
            self.assertTrue(CH[child_indeces[i],0]==-1) 
            self.assertTrue(CH[child_indeces[i],1]==-1) 
            self.assertTrue(CH[child_indeces[i],2]==-1) 
            self.assertTrue(CH[child_indeces[i],3]==-1) 




if __name__ == '__main__':
    unittest.main()
