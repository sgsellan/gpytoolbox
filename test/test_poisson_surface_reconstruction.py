from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

# Would be nice to get rid of this

class TestPoissonSurfaceReconstruction(unittest.TestCase):
    def test_indicator(self):
        np.random.seed(0)
        # First test: "uniform" sampling density
        # Sample points on a circle
        th = 2*np.pi*np.random.rand(500,1)
        P = np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Normals are the same as positions on a circle
        N = np.concatenate((np.cos(th),np.sin(th)),axis=1)

        corner = np.array([-1.5,-1.5])
        gs = np.array([305, 302])
        h = np.array([0.01,0.01])

        g, sigma = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,h=h,corner=corner,screened=True)

        #print(sigma)


        # Sample random points inside a circle
        th = 2*np.pi*np.random.rand(1000,1)
        points_inside_circle = 0.98*np.random.rand(1000,2)*np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Interpolate value of g at these points
        W = gpytoolbox.fd_interpolate(points_inside_circle,gs=gs,h=h,corner=corner)
        values = W @ (g - sigma)
        # All these values should be negative (they are inside)
        self.assertTrue((values<0).all())
        # Sample random points outside the circle (radius < 1.5 so we are inside grid)
        th = 2*np.pi*np.random.rand(1000,1)
        points_outside_circle = (1.02 + 0.2*np.random.rand(1000,2))*np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Interpolate value of g at these points
        W = gpytoolbox.fd_interpolate(points_outside_circle,gs=gs,h=h,corner=corner)
        values = W @ (g - sigma)
        # All these values should be positive (they are outside)
        self.assertTrue((values>0).all())

        # For visualization, use ij indexing
        x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),indexing='ij')
        #V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        # plt.pcolor(x, y, np.reshape(g - sigma,x.shape,order='F'),shading='auto')
        # plt.colorbar()
        # plt.title("Does this look like a circle?")
        # plt.show(block=False)

        # plt.pause(20)

        # plt.close()

        # Second test: distinctly non-uniform sampling density

        # Generate many more on top than on bottom
        th = np.vstack(( np.pi*np.random.rand(50,1), np.pi + np.pi*np.random.rand(500,1) ))
        P = np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Normals are the same as positions on a circle
        N = np.concatenate((np.cos(th),np.sin(th)),axis=1)
        # PSR should weigh by the local sampling density!
        # N[500:5500,:] = 0.1*N[500:5500,:]

        corner = np.array([-1.5,-1.5])
        gs = np.array([305, 302])
        h = np.array([0.01,0.01])

        g, sigma = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,h=h,corner=corner,screened=True)

        #print(sigma)


        # Sample random points inside a circle
        th = 2*np.pi*np.random.rand(1000,1)
        points_inside_circle = 0.98*np.random.rand(1000,2)*np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Interpolate value of g at these points
        W = gpytoolbox.fd_interpolate(points_inside_circle,gs=gs,h=h,corner=corner)
        values = W @ (g - sigma)
        # All these values should be negative (they are inside)
        self.assertTrue((values<0).all())
        # Sample random points outside the circle (radius < 1.5 so we are inside grid)
        th = 2*np.pi*np.random.rand(1000,1)
        points_outside_circle = (1.02 + 0.2*np.random.rand(1000,2))*np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Interpolate value of g at these points
        W = gpytoolbox.fd_interpolate(points_outside_circle,gs=gs,h=h,corner=corner)
        values = W @ (g - sigma)
        # All these values should be positive (they are outside)
        self.assertTrue((values>0).all())

        # For visualization, use ij indexing
        x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),indexing='ij')
        #V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        # plt.pcolor(x, y, np.reshape(g - sigma,x.shape,order='F'),shading='auto')
        # plt.colorbar()
        # plt.title("Does this look like a circle?")
        # plt.show(block=False)
        # plt.pause(20)
        # #plt.clf()
        # plt.close()
        # print("Unit test passed, all self.assertTrues passed")

if __name__ == '__main__':
    unittest.main()