from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
from numpy.random import default_rng

class TestArrayCorrespondence(unittest.TestCase):

    def test_array1(self):
        A = np.array([1,20,1,4,41,30,15,1])
        B = np.array([20,1,1,4,1,4,1,0,1,30,7])
        f = gpy.array_correspondence(A,B)
        self.mapping_condition(A, B, f)
        fi = gpy.array_correspondence(B,A)
        self.mapping_condition(B, A, fi)

    def test_random_arrays(self):
        rng = default_rng()
        dims = rng.integers(1, 10000, size=(20,2))
        for i in range(dims.shape[0]):
            A = rng.integers(0, 10000, size=dims[i,0])
            B = rng.integers(0, 10000, size=dims[i,1])
            f = gpy.array_correspondence(A,B)
            self.mapping_condition(A, B, f)
            fi = gpy.array_correspondence(B,A)
            self.mapping_condition(B, A, fi)

    def mapping_condition(self, A, B, f):
        self.assertTrue(f.shape[0] == A.shape[0])
        listA = A.tolist()
        listB = B.tolist()
        for i,m in enumerate(f.tolist()):
            if m<0:
                #Claim: this element is in A, but not in B
                self.assertTrue(listA[i] not in listB)
            else:
                self.assertTrue(listA[i]==listB[m])

if __name__ == '__main__':
    unittest.main()

