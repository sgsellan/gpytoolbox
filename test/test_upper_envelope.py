from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
# Would be nice to get rid of this
# print("0")
import tetgen

# TO DO: Expand this.
class TestUpperEnvelope(unittest.TestCase):
    def test_no_inversions(self):
        self.assertTrue(True)
        # Load mesh
        # print("1")
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # Generate many examples
        for m in np.linspace(20,5000,20,dtype=int):
            # print("2")
            v,f = gpytoolbox.copyleft.lazy_cage(v,f,num_faces=m)
            # print("3")
            v = gpytoolbox.normalize_points(v)
            # print("4")
            tgen = tetgen.TetGen(v,f)
            # print("5")
            v, t =  tgen.tetrahedralize()
            # print("6")
            d = np.zeros((v.shape[0],2))
            d[:,0] = 1.0 - v[:,2]
            d[:,1] = 1.0 - v[:,1]
            ## print(igl.volume(v,t))
            u, g, l = gpytoolbox.upper_envelope(v,t,d)
            # print("7")
            # self.assertTrue that no tet is flipped:
            self.assertTrue(np.min(gpytoolbox.volume(u,g))>-1e-8)
    def test_issue_143(self):
        labels = 4
        resolution = 4 # The issue said 5, but I needed to make it 4 to break it
        v, t = gpytoolbox.regular_cube_mesh(resolution)
        w = np.zeros((v.shape[0],labels))
        for i in range(labels):
            p = np.random.rand(3)
            # print( "Center: ", p )
            for j in range(v.shape[0]):
                w[j,i] = -np.linalg.norm( v[j] - p )
        
        # This used to crash
        u, g, l = gpytoolbox.upper_envelope(v,t,w)
        # but it doesn't anymore

if __name__ == '__main__':
    unittest.main()