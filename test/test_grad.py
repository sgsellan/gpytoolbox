from gpytoolbox.edge_indeces import edge_indeces
from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import igl
from scipy.sparse import csr_matrix

class TestGrad(unittest.TestCase):
    def test_polyline_grad(self):
        # This is a cube, centered at the origin, with side length 1
        # v,f = igl.read_triangle_mesh("test/unit_tests_data/cube.obj")
        #
        # Let's make up a simple polyline
        v = np.array([[0],[0.2],[0.5],[0.98],[1.0]])
        edge_centers = (v[0:4,:] + v[1:5,:])/2.0
        fun_zero_grad = 0*v + 5
        fun_constant_grad = 2*v
        fun_other_grad = v**2.0
        G = gpytoolbox.grad(v)
        # Finite elements should get exact gradients in these
        self.assertTrue(np.isclose((G @ fun_zero_grad),0.0).all())
        self.assertTrue(np.isclose((G @ fun_constant_grad) - 2.0,0.0).all())
        self.assertTrue(np.isclose((G @ fun_other_grad) - 2.0*edge_centers,0.0).all())

    def test_single_triangle_2d(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        G = gpytoolbox.grad(v,f)
        G_gt = np.array([[-1.0,1.0,0.0],[-1,0.0,1.0]])
        #print(G)
        #print(csr_matrix(igl.grad(np.hstack((v,np.zeros((v.shape[0],1)))),f)))
        self.assertTrue(np.isclose(G.toarray() - G_gt,0.0).all())
        # print(G-igl.grad(np.hstack((v,np.zeros((v.shape[0],1)))),f))
    
    def test_single_triangle_3d(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        G = gpytoolbox.grad(v,f)
        G_gt = np.array([[-1.0,1.0,0.0],[0.0,0.0,0.0],[-1,0.0,1.0]])
        self.assertTrue(np.isclose(G.toarray() - G_gt,0.0).all())
        # print(G-igl.grad(np.hstack((v,np.zeros((v.shape[0],1)))),f))

    def test_2d_grad(self):
        v,f = gpytoolbox.regular_square_mesh(400)
        barycenters = (v[f[:,0],:] + v[f[:,1],:] + v[f[:,2],:])/3.0

        fun_zero_grad = 0*v[:,0] + 5
        fun_constant_grad = v[:,0] + 2*v[:,1]
        fun_other_grad = v[:,0]**2.0 + 5*v[:,1]**2.0 + 3.0*v[:,0]
        G = gpytoolbox.grad(v,f)
        # Finite elements should get exact gradients if they are analytically piecewise linear
        self.assertTrue(np.isclose((G @ fun_zero_grad),0.0).all())
        self.assertTrue(np.isclose((G @ fun_constant_grad)[0:f.shape[0]] - 1.0,0.0).all())
        self.assertTrue(np.isclose((G @ fun_constant_grad)[f.shape[0]:(2*f.shape[0])] - 2.0,0.0).all())
        # For a function without constant gradients it shouldn't be exact but it should converge
        err = 1.0
        for i in range(2,10):
            v,f = gpytoolbox.regular_square_mesh(2**i)
            barycenters = (v[f[:,0],:] + v[f[:,1],:] + v[f[:,2],:])/3.0
            fun_other_grad = v[:,0]**2.0 + 5*v[:,1]**2.0 + 3.0*v[:,0]
            G = gpytoolbox.grad(v,f)
            self.assertTrue(np.amax(np.abs((G @ fun_other_grad)[0:f.shape[0]] - (2.0*barycenters[:,0] + 3)))<err )
            err = np.amax(np.abs((G @ fun_other_grad)[0:f.shape[0]] - (2.0*barycenters[:,0] + 3)))

    def test_3d_grad(self):
        v,f = igl.read_triangle_mesh("test/unit_tests_data/bunny_oded.obj")
        #v = v[:,0:2]
        G = gpytoolbox.grad(v,f).toarray()
        G_igl = csr_matrix(igl.grad(v,f)).toarray()
        self.assertTrue(np.isclose(G-G_igl,0.0).all())

        


if __name__ == '__main__':
    unittest.main()