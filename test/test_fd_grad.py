from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

# This test is basically the same as fd_partial_derivative

class TestFdGrad(unittest.TestCase):
    def test_analytic_2d(self):
        # Choose grid size
        gs = np.array([19,15])
        h = 1.0/(gs-1)

        # Build a grid
        x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]))
        V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

        # Build staggered grid in x direction
        x, y = np.meshgrid(np.linspace(0+0.5*h[0],1-0.5*h[0],gs[0]-1),np.linspace(0,1,gs[1]))
        Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        # Build staggered grid in y direction
        x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0+0.5*h[1],1-0.5*h[1],gs[1]-1))
        Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

        # Compute gradient
        G = gpytoolbox.fd_grad(gs=gs,h=h)


        # all rows must sum up to zero (i.e. a constant function has zero derivative)
        self.assertTrue((np.isclose(G.sum(axis=1),np.zeros((G.shape[0],1)))).all())

        # Build linear function
        f = 2*V[:,0] + 5*V[:,1]
        computed_grad = G*f
        computed_derivative_x = computed_grad[0:(gs[1]*(gs[0]-1))] # staggered x
        computed_derivative_y = computed_grad[(gs[1]*(gs[0]-1)):((gs[1]*(gs[0]-1)) + (gs[0]*(gs[1]-1))) ]
        # Derivatives must be 2.0 and 5.0, respectively
        self.assertTrue((np.isclose(computed_derivative_x,2.0*np.ones((computed_derivative_x.shape[0])))).all())
        self.assertTrue((np.isclose(computed_derivative_y,5.0*np.ones((computed_derivative_y.shape[0])))).all())#self.assertTrue((np.isclose(computed_derivative_y,5.0*np.ones((computed_derivative_y.shape[0])))).all())
    def test_convergence_2d(self):
        # Convergence test
        linf_norm_x = 100.0
        linf_norm_y = 100.0
        # print("This experiment should # print a set of decreasing values, converging")
        # print("towards zero and decreasing roughly by half in each iteration")
        for power in range(3,13,1):
            gs = np.array([2**power,2**power - 2])
            h = 1.0/(gs-1)
            # Build a grid
            x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]))
            V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

            # Build staggered grid in x direction
            x, y = np.meshgrid(np.linspace(0+0.5*h[0],1-0.5*h[0],gs[0]-1),np.linspace(0,1,gs[1]))
            Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
            # Build staggered grid in y direction
            x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0+0.5*h[1],1-0.5*h[1],gs[1]-1))
            Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
            # Build derivative matrices
            G = gpytoolbox.fd_grad(gs=gs,h=h)
            # Build non-linear function
            f = np.cos(V[:,0]) + np.sin(V[:,1])
            # Derivatives on staggered grids
            fx = -np.sin(Vx[:,0])
            fy =  np.cos(Vy[:,1])
            # Computed derivatives using our matrices
            computed_grad = G*f
            computed_derivative_x = computed_grad[0:(gs[1]*(gs[0]-1))] # staggered x
            computed_derivative_y = computed_grad[(gs[1]*(gs[0]-1)):((gs[1]*(gs[0]-1)) + (gs[0]*(gs[1]-1))) ]
            # Make sure norm is decreasing
            self.assertTrue(linf_norm_x>np.max(np.abs(computed_derivative_x - fx)))
            self.assertTrue(linf_norm_y>np.max(np.abs(computed_derivative_y - fy)))
            linf_norm_x = np.max(np.abs(computed_derivative_x - fx))
            linf_norm_y = np.max(np.abs(computed_derivative_y - fy))
            # # print L infinity norm of difference
            # print(np.array([np.max(np.abs(computed_derivative_x - fx)),np.max(np.abs(computed_derivative_y - fy))]))

    def test_analytic_3d(self):
        # 3D Unit test
        # Choose grid size
        gs = np.array([19,15,23])
        h = 1.0/(gs-1)

        # Build a grid
        x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),np.linspace(0,1,gs[2]),indexing='ij')
        V = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)

        # Build staggered grid in x direction
        x, y, z = np.meshgrid(np.linspace(0+0.5*h[0],1-0.5*h[0],gs[0]-1),np.linspace(0,1,gs[1]),np.linspace(0,1,gs[2]),indexing='ij')
        Vx = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)
        # Build staggered grid in y direction
        x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0+0.5*h[1],1-0.5*h[1],gs[1]-1),np.linspace(0,1,gs[2]),indexing='ij')
        Vy = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)
        # Build staggered grid in z direction
        x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),np.linspace(0+0.5*h[2],1-0.5*h[2],gs[2]-1),indexing='ij')
        Vz = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)

        # Compute gradient
        G = gpytoolbox.fd_grad(gs=gs,h=h)

        # Build linear function
        f = 2*V[:,0] + 5*V[:,1] - 4*V[:,2]
        computed_grad = G*f
        computed_derivative_x = computed_grad[0:(gs[1]*(gs[0]-1)*gs[2])] # staggered x
        computed_derivative_y = computed_grad[(gs[1]*(gs[0]-1)*gs[2]):((gs[1]*(gs[0]-1)*gs[2]) + (gs[0]*(gs[1]-1)*gs[2])) ]
        computed_derivative_z = computed_grad[((gs[1]*(gs[0]-1)*gs[2]) + (gs[0]*(gs[1]-1)*gs[2])):((gs[1]*(gs[0]-1)*gs[2]) + (gs[0]*(gs[1]-1)*gs[2]) + (gs[0]*(gs[2]-1)*gs[1])) ]
        # Derivatives must be 2.0 and 5.0, respectively
        self.assertTrue((np.isclose(computed_derivative_x,2.0*np.ones((computed_derivative_x.shape[0])))).all())
        self.assertTrue((np.isclose(computed_derivative_y,5.0*np.ones((computed_derivative_y.shape[0])))).all())
        self.assertTrue((np.isclose(computed_derivative_z,-4.0*np.ones((computed_derivative_z.shape[0])))).all())

    def test_convergence_3d(self):
        # Convergence test
        linf_norm_x = 100.0
        linf_norm_y = 100.0
        linf_norm_z = 100.0
        # print("This experiment should # print a set of decreasing values, converging")
        # print("towards zero and decreasing roughly by half in each iteration")
        for power in range(3,9,1):
            gs = np.array([2**power,2**power - 2,2**power - 1])
            h = 1.0/(gs-1)
            # Build a grid
            x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),np.linspace(0,1,gs[2]),indexing='ij')
            V = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)

            # Build staggered grid in x direction
            x, y, z = np.meshgrid(np.linspace(0+0.5*h[0],1-0.5*h[0],gs[0]-1),np.linspace(0,1,gs[1]),np.linspace(0,1,gs[2]),indexing='ij')
            Vx = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)
            # Build staggered grid in y direction
            x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0+0.5*h[1],1-0.5*h[1],gs[1]-1),np.linspace(0,1,gs[2]),indexing='ij')
            Vy = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)
            # Build staggered grid in z direction
            x, y, z = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]),np.linspace(0+0.5*h[2],1-0.5*h[2],gs[2]-1),indexing='ij')
            Vz = np.concatenate((np.reshape(x,(-1, 1),order='F'),np.reshape(y,(-1, 1),order='F'),np.reshape(z,(-1, 1),order='F')),axis=1)

            # Compute gradient
            G = gpytoolbox.fd_grad(gs=gs,h=h)


            # Build non-linear function
            f = np.cos(V[:,0]) + np.sin(V[:,1]) + 3*np.cos(V[:,2])
            # Derivatives on staggered grids
            fx = -np.sin(Vx[:,0])
            fy =  np.cos(Vy[:,1])
            fz =-3*np.sin(Vz[:,2])
            computed_grad = G*f
            computed_derivative_x = computed_grad[0:(gs[1]*(gs[0]-1)*gs[2])] # staggered x
            computed_derivative_y = computed_grad[(gs[1]*(gs[0]-1)*gs[2]):((gs[1]*(gs[0]-1)*gs[2]) + (gs[0]*(gs[1]-1)*gs[2])) ]
            computed_derivative_z = computed_grad[((gs[1]*(gs[0]-1)*gs[2]) + (gs[0]*(gs[1]-1)*gs[2])):((gs[1]*(gs[0]-1)*gs[2]) + (gs[0]*(gs[1]-1)*gs[2]) + (gs[0]*(gs[2]-1)*gs[1])) ]
            self.assertTrue(linf_norm_x>np.max(np.abs(computed_derivative_x - fx)))
            self.assertTrue(linf_norm_y>np.max(np.abs(computed_derivative_y - fy)))
            self.assertTrue(linf_norm_z>np.max(np.abs(computed_derivative_z - fz)))
            linf_norm_x = np.max(np.abs(computed_derivative_x - fx))
            linf_norm_y = np.max(np.abs(computed_derivative_y - fy))
            linf_norm_z = np.max(np.abs(computed_derivative_z - fz))
            # # print L infinity norm of difference
            # print(np.array([np.max(np.abs(computed_derivative_x - fx)),np.max(np.abs(computed_derivative_y - fy)),np.max(np.abs(computed_derivative_z - fz))]))


if __name__ == '__main__':
    unittest.main()