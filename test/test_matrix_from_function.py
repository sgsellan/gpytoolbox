from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestMatrixFromFunction(unittest.TestCase):
    def test_random_arrays(self):
        def sample_fun(X1,X2):
            # Make it be something non-symmetric just to be sure
            return np.linalg.norm(X1-2*X2,axis=1)    
        # Dimension shouldn't matter  
        for dd in range(1,5):
            P1 = np.random.rand(50,dd)
            P2 = np.random.rand(71,dd)
            # Assert with sparse matrix
            M = gpytoolbox.matrix_from_function(sample_fun,P1,P2,sparse=True)
            for i in range(P1.shape[0]):
                for j in range(P2.shape[0]):
                    self.assertTrue(M[i,j]==sample_fun(P1[i,:][None,:],P2[j,:][None,:]))
            # Assert with dense matrix
            M = gpytoolbox.matrix_from_function(sample_fun,P1,P2,sparse=False)
            for i in range(P1.shape[0]):
                for j in range(P2.shape[0]):
                    self.assertTrue(M[i,j]==sample_fun(P1[i,:][None,:],P2[j,:][None,:]))
            

if __name__ == '__main__':
    unittest.main()

