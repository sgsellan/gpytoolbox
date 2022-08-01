from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestDecimate(unittest.TestCase):
    def test_armadillo(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/armadillo.obj")
        for nn in range(20,2000,301):
            u,g,i,j = gpytoolbox.decimate(v,f,num_faces=nn)
            self.assertTrue(np.isclose(g.shape[0]-nn,0,atol=3))
            ratio = nn/f.shape[0]
            u,g,i,j = gpytoolbox.decimate(v,f,face_ratio=ratio)
            # print(nn)
            # print(g.shape[0])
            # print(g.shape[0]/f.shape[0])
            self.assertTrue(np.isclose(ratio - (g.shape[0]/f.shape[0]),0,atol=0.001))
            # Are the outputs what they claim they are?
            # Is i the size of g and j the size of u?
            self.assertTrue(g.shape[0]==i.shape[0])
            self.assertTrue(u.shape[0]==j.shape[0])
            # There isn't really a good way to check that one is the birth vertex of the other...

    def test_with_boundary(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/airplane.obj")
        for nn in range(200,2000,301):
            u,g,i,j = gpytoolbox.decimate(v,f,num_faces=nn)
            self.assertTrue(np.isclose(g.shape[0]-nn,0,atol=3))
            ratio = nn/f.shape[0]
            u,g,i,j = gpytoolbox.decimate(v,f,face_ratio=ratio)
            self.assertTrue(np.isclose(ratio - (g.shape[0]/f.shape[0]),0,atol=0.001))
            gpytoolbox.write_mesh("output.obj",u,g)
            # Are the outputs what they claim they are?
            # Is i the size of g and j the size of u?
            self.assertTrue(g.shape[0]==i.shape[0])
            self.assertTrue(u.shape[0]==j.shape[0])
            # There isn't really a good way to check that one is the birth vertex of the other...
        
if __name__ == '__main__':
    unittest.main()