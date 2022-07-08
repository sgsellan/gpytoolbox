from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestFdInterpolate(unittest.TestCase):
    def test_analytic_2d(self):
        #2D unit testing
        gs = np.array([19,15])
        h = 1.0/(gs-1)
        for iter in range(1,100,1):
            # Build a grid
            x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]))
            V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
            # Random set of points
            P = np.random.rand(10,2)

            # Random grid corner
            corner = np.random.rand(2)

            # Displace by corner
            P = P + np.tile(corner,(P.shape[0],1))
            V = V + np.tile(corner,(V.shape[0],1))

            W = gpytoolbox.fd_interpolate(P,gs=gs,h=h,corner=corner)
            # all rows must sum up to one
            self.assertTrue((np.isclose(W.sum(axis=1),np.ones((W.shape[0],1)))).all())

            # Does W do what it says it does? Can it interpolate the grid positions?
            self.assertTrue((np.isclose(P,W @ V)).all())

            # Choose a linear function f = 3*x + 5*y
            # Bilinear interpolation should exactly compute this
            fP = 3.0*P[:,0] + 5.0*P[:,1]
            fgrid = 3.0*V[:,0] + 5.0*V[:,1]
            finterp = W @ fgrid
            self.assertTrue((np.isclose(fP,finterp)).all())

    def test_analytic_3d(self):
        #3D unit testing
        gs = np.array([3,2,4])
        h = 1.0/(gs-1)
        for iter in range(1,100,1):
            # Build a grid
            x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),np.linspace(0,1,gs[2]),indexing='ij')
            V = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)
            # Random set of points
            P = np.random.rand(10,3)

            # Random grid corner
            corner = np.random.rand(3)

            # Displace by corner
            P = P + np.tile(corner,(P.shape[0],1))
            V = V + np.tile(corner,(V.shape[0],1))

            W = gpytoolbox.fd_interpolate(P,gs=gs,h=h,corner=corner)
            # all rows must sum up to one
            self.assertTrue((np.isclose(W.sum(axis=1),np.ones((W.shape[0],1)))).all())

            # Does W do what it says it does? Can it interpolate the grid positions?
            self.assertTrue((np.isclose(P,W @ V)).all())

            # Choose a linear function f = 3*x + 5*y
            # Bilinear interpolation should exactly compute this
            fP = 3.0*P[:,0] + 5.0*P[:,1] + 7.0*P[:,2]
            fgrid = 3.0*V[:,0] + 5.0*V[:,1] + 7.0*V[:,2]
            finterp = W @ fgrid
            self.assertTrue((np.isclose(fP,finterp)).all())

if __name__ == '__main__':
    unittest.main()