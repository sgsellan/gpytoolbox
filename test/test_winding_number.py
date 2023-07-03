import numpy as np
from .context import gpytoolbox
from .context import unittest
import matplotlib.pyplot as plt

class TestWindingNumber(unittest.TestCase):
    # This isn't too complex, probably could use being expanded
    def test_synthetic(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])
        EC = gpytoolbox.edge_indices(V.shape[0],closed=True)
        # 2d grid
        gx, gy = np.meshgrid(np.linspace(-2.01,2.01,100),np.linspace(-2.01,2.01,100))
        gv = np.vstack((gx.flatten(),gy.flatten())).T
        # Compute winding number
        W = gpytoolbox.winding_number(gv,V,EC)

        # groundtruth winding number: 1 inside, 0 outside
        W_gt = np.zeros(gv.shape[0])
        W_gt[(gv[:,0]>-1) & (gv[:,0]<1) & (gv[:,1]>-1) & (gv[:,1]<1)] = 1
        # assert np.allclose(W,W_gt)
        self.assertTrue(np.allclose(W,W_gt))

        # Let's add another square intersecting this sqaure
        Vbig = np.vstack((V, V + 1.0))
        ECbig = np.vstack((EC, EC + V.shape[0]))
        W = gpytoolbox.winding_number(gv,Vbig,ECbig)

        # groundtruth
        W_gt = np.zeros(gv.shape[0])
        W_gt[(gv[:,0]>-1) & (gv[:,0]<1) & (gv[:,1]>-1) & (gv[:,1]<1)] = 1
        W_gt[(gv[:,0]>0) & (gv[:,0]<2) & (gv[:,1]>0) & (gv[:,1]<2)] = 1
        # when the two squares intersect, the winding number is 2
        W_gt[(gv[:,0]>0) & (gv[:,0]<1) & (gv[:,1]>0) & (gv[:,1]<1)] = 2
        # self.assertTrue(np.allclose(np))
        self.assertTrue(np.allclose(W,W_gt))

    def test_meshes(self):
        meshes = ["bunny_oded.obj", "cube.obj"]
        bools = [True, False]
        for mesh in meshes:
            for bbool in bools:
                num_points = 1000
                V,F = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
                # Generate random points on mesh
                Q,I,_ = gpytoolbox.random_points_on_mesh(V,F,num_points,return_indices=True,rng=np.random.default_rng(5))
                # Per face normals
                N = gpytoolbox.per_face_normals(V,F,unit_norm=True)
                # Compute winding number
                eps = 1e-3
                points_out = Q + eps*N[I,:]
                points_in = Q - eps*N[I,:]
                w_in = gpytoolbox.winding_number(points_in,V,F)
                w_out = gpytoolbox.winding_number(points_out,V,F)
                # print(wn_in)
                # print(wn_out)
                # print(np.isclose(wn_out,0,atol=1e-2))
                self.assertTrue(np.isclose(w_out,0.0,atol=1e-2).all())
                self.assertTrue(np.isclose(w_in,1.0,atol=1e-2).all())
        

if __name__ == '__main__':
    unittest.main()
