from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestQuadtreeGradient(unittest.TestCase):
    def test_error_is_low(self):
        np.random.seed(0)
        th = 2*np.pi*np.random.rand(100,1)
        P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)

        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=6,vmin=np.array([-1,-1]),vmax=np.array([1,1]))


        G, stored_at = gpytoolbox.quadtree_gradient(C,W,CH,D,A)
        Gx = G[0:stored_at.shape[0],:]
        Gy = G[stored_at.shape[0]:(2*stored_at.shape[0]),:]
        fun = stored_at[:,0]
        # This should be one everywhere
        self.assertTrue(np.all(np.isclose(Gx @ stored_at[:,0],1.0)))
        # This will never be exactly zero (unless you subdivide everything at once), but it should be zero "in most places"
        self.assertTrue(np.isclose(np.median(np.abs(Gy @ stored_at[:,0])),0.0))
        # print("Unit test passed, all self.assertTrues passed")

        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=5,vmin=np.array([-1,-1]),vmax=np.array([1,1]))

        G, stored_at = gpytoolbox.quadtree_gradient(C,W,CH,D,A)
        Gx = G[0:stored_at.shape[0],:]
        Gy = G[stored_at.shape[0]:(2*stored_at.shape[0]),:]
        fun = stored_at[:,0]**2.0
        dx = 2*stored_at[:,0]
        # print(np.mean(np.abs(Gx @ fun - dx)))
        # It is really hard to conclude what an acceptable error is, since there are many parts of the grid that are very coarse. Let's say a mean under 0.01 is acceptable, but since this is all a hack it's hard to justify. This should at least catch major breaks in the code
        self.assertTrue(np.isclose(np.median(np.abs(Gx @ fun - dx)),0.0,atol=1e-2))

    def test_convergence(self):
        # A more precise way of testing convergence is by progressively adding points, then the error should strictly go down (the grid gets finer)
        np.random.seed(0)
        err = 0.1
        for i in range(2,10):
            P = 2 * np.random.rand(2**i,2) - 1
            C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=1,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
            V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)


            G, stored_at = gpytoolbox.quadtree_gradient(C,W,CH,D,A)
            Gx = G[0:stored_at.shape[0],:]
            Gy = G[stored_at.shape[0]:(2*stored_at.shape[0]),:]
            fun = stored_at[:,0]**2.0
            dx = 2*stored_at[:,0]
            # Check that it got better
            self.assertTrue(err > np.mean(np.abs(Gx @ fun - dx)))
            err = np.mean(np.abs(Gx @ fun - dx))

if __name__ == '__main__':
    unittest.main()
