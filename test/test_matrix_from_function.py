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

    def test_sparsity_pattern(self):
        def sample_fun(X1,X2):
            # Make it be something non-symmetric just to be sure
            return np.linalg.norm(X1-2*X2,axis=1)    
        # Dimension shouldn't matter  
        for dd in range(1,5):
            P1 = np.random.rand(50,dd)
            P2 = np.random.rand(75,dd)
            # Assert with sparse matrix
            # Pick some indices
            I = np.arange(0,50,2)
            J = np.arange(0,75,3)
            sparsity_pattern = [I,J]
            M = gpytoolbox.matrix_from_function(sample_fun,P1,P2,sparse=True,sparsity_pattern=sparsity_pattern)
            for i in range(P1.shape[0]):
                # Find index of i in I
                ind_i = np.where(I==i)[0]
                # print(ind_i)
                if ind_i.size>0:
                    for j in range(P2.shape[0]):
                        if J[ind_i]==j:
                            # print(sample_fun(P1[i,:][None,:],P2[j,:][None,:]))
                            self.assertTrue(M[i,j]==sample_fun(P1[i,:][None,:],P2[j,:][None,:]))
                        else:
                            self.assertTrue(M[i,j]==0)      
                else:
                    self.assertTrue(M[i,j]==0)      
        # Define compactly supported sample function
        def sample_fun(X1,X2):
            # Make it be something non-symmetric just to be sure
            d = np.linalg.norm(X1-2*X2,axis=1)
            d[d>0.1] = 0
            return d
        # Dimension shouldn't matter
        for dd in range(1,5):
            P1 = np.random.rand(50,dd)
            P2 = np.random.rand(75,dd)
            I = []
            J = []
            for i in range(P1.shape[0]):
                for j in range(P2.shape[0]):
                    if sample_fun(P1[i,:][None,:],P2[j,:][None,:])>0.0:
                        I.append(i)
                        J.append(j)
            sparsity_pattern = [I,J]
            M = gpytoolbox.matrix_from_function(sample_fun,P1,P2,sparse=True,sparsity_pattern=sparsity_pattern)
            # Then, since we've designed the sparsity pattern so that it only includes nonzero entries, this should return the same matrix but having visited less entries
            for i in range(P1.shape[0]):
                for j in range(P2.shape[0]):
                    self.assertTrue(M[i,j]==sample_fun(P1[i,:][None,:],P2[j,:][None,:]))

if __name__ == '__main__':
    unittest.main()

