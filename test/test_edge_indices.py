import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestEdgeIndices(unittest.TestCase):
    def test_indices_make_sense(self):
        for nn in range(3,200,20):
            EC = gpytoolbox.edge_indices(nn)
            self.assertTrue(EC.shape[0]==(nn-1))
            self.assertTrue((EC[:,0]==(EC[:,1]-1)).all())
            EC = gpytoolbox.edge_indices(nn,closed=True)
            self.assertTrue(EC.shape[0]==nn)
            self.assertTrue((EC[0:(nn-1),0]==(EC[0:(nn-1),1]-1)).all())


if __name__ == '__main__':
    unittest.main()
