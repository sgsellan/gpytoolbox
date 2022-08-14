from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestInitializeAabbTree(unittest.TestCase):
    def test_simple_groundtruth(self):
        np.random.seed(0)
        th = 2*np.pi*np.random.rand(200,1)
        P = np.array([[0,0,0],[0.1,0,0],[0,0,0],[0.02,0.1,0],[1,1,0.9],[1,1,1]])
        F = np.array([[0,1],[2,3],[4,5]],dtype=np.int32)
        C,W,CH,PAR,D,tri_ind = gpytoolbox.initialize_aabbtree(P,F)
        C_gt = np.array([[0.5,  0.5,  0.5 ],
            [0.05, 0.05, 0.,  ],
            [1.,   1.,   0.95,],
            [0.01, 0.05, 0.,  ],
            [0.05, 0.,   0.,  ]])
        W_gt = np.array([[1.,   1.,   1.,  ],
            [0.1,  0.1,  0.  ],
            [0.,   0.,   0.1 ],
            [0.02, 0.1,  0.  ],
            [0.1,  0.,   0.  ]])
        tri_ind_gt = np.array([[-1],
            [-1],
            [ 2],
            [ 1],
            [ 0]])
        CH_gt = np.array([[ 1,  2],
            [ 3,  4],
            [-1, -1],
            [-1, -1],
            [-1, -1]])
        PAR_gt = np.array([-1,  0,  0,  1,  1])
        self.assertTrue((np.isclose(C-C_gt,0).all()))
        self.assertTrue((np.isclose(W-W_gt,0).all()))
        self.assertTrue((np.isclose(tri_ind-tri_ind_gt,0).all()))
        self.assertTrue((np.isclose(CH-CH_gt,0).all()))
        self.assertTrue((np.isclose(PAR-PAR_gt,0).all()))

    def test_consistency(self):
        np.random.seed(0)
        for ss in range(10,20000,100):       
            P = np.random.rand(11,2)
            ptest = P[9,:] + 1e-5
            C,W,CH,PAR,D,tri_ind = gpytoolbox.initialize_aabbtree(P)
            # Parenthood stuff
            for i in range(W.shape[0]):
                for ss in range(CH.shape[1]):
                    if CH[i,ss]!=-1:
                        self.assertTrue(PAR[CH[i,ss]]==i)
                # The parent of i must have i as a child
                if PAR[i]>0: #We are not in the supreme dad node
                    self.assertTrue(i in CH[PAR[i],:])
            # Now, for every point in P, there must be one that contains it as tri_ind
            for i in range(P.shape[0]):
                self.assertTrue(i in tri_ind)




if __name__ == '__main__':
    unittest.main()
