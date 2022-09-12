from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
# import igl
# from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# import polyscope as ps


class TestCompactlySupportedNormal(unittest.TestCase):
    def test_1d(self):
        np.random.seed(0)
        num_samples = 10000
        x = np.reshape(np.linspace(-8,8,num_samples),(-1,1))
        # x = np.linspace(-8,8,num_samples)
        for oo in range(1,5):
            rand_center = 2*np.random.rand(1)-1
            rand_scale = np.random.rand(1)
            v = gpytoolbox.compactly_supported_normal(x, n=oo, center=rand_center,sigma=rand_scale)
            calculated_center = x[np.argmax(v)]
            # Center is the center
            self.assertTrue(np.isclose(rand_center-calculated_center,0.0,atol=1e-3))
            before_center = v[0:np.argmax(v)]
            after_center = v[np.argmax(v):num_samples]
            # Assert is monotonous
            self.assertTrue(np.all(before_center[:-1] <= before_center[1:]))
            self.assertTrue(np.all(after_center[:-1] >= after_center[1:]))
            # Assert is compact
            self.assertTrue(v[0]==0.0)
            self.assertTrue(v[num_samples-1]==0.0)
            # Now check derivatives:
            dx = x[1]-x[0]
            center = x[1:(num_samples-1)]
            fd = (-2*v[1:(num_samples-1)] + v[2:num_samples] + v[0:(num_samples-2)] )/(dx**2.0)
            fun_der = gpytoolbox.compactly_supported_normal(center, n=oo, center=rand_center,sigma=rand_scale,second_derivative=0)
            # There are discontinuities in second derivative
            self.assertTrue(np.mean(fun_der-fd)<0.001)
            # Integral is one:
            self.assertTrue(np.isclose(np.sum(v)*dx - 1,0.0,atol=1e-3))
    def test_2d(self):
        np.random.seed(0)
        num_samples = 2000
        x, y = np.meshgrid(np.linspace(-8,8,num_samples),np.linspace(-8,8,num_samples))
        V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        h = x[1,1] - x[0,0]
        h = np.array([h,h])
        x, y = np.meshgrid(np.linspace(-8+h[0],8-h[0],num_samples-2),np.linspace(0,1,num_samples))
        Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        # Build staggered grid in y direction
        x, y = np.meshgrid(np.linspace(0,1,num_samples),np.linspace(8+h[1],8-h[1],num_samples-2))
        Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        
        for oo in range(1,5):
            rand_center = 2*np.random.rand(2)-1
            rand_scale = np.random.rand(1)
            v = gpytoolbox.compactly_supported_normal(V, n=oo, center=rand_center,sigma=rand_scale)
            calculated_center = V[np.argmax(v),:]
            # Center is the center
            self.assertTrue(np.isclose(rand_center-calculated_center,0.0,atol=1e-2).all())
            # Finite difference estimation
            W1 = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,0)
            W2 = gpytoolbox.fd_partial_derivative(np.array([num_samples-1,num_samples]),h,0)
            fd_x = W2 @ W1 @ v
            W1 = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
            W2 = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples-1]),h,1)
            fd_y = W2 @ W1 @ v
            fun_der_x = gpytoolbox.compactly_supported_normal(Vx, n=oo, center=rand_center,sigma=rand_scale,second_derivative=0)
            fun_der_y = gpytoolbox.compactly_supported_normal(Vy, n=oo, center=rand_center,sigma=rand_scale,second_derivative=0)
            self.assertTrue(np.mean(fun_der_x-fd_x)<0.001)
            self.assertTrue(np.mean(fun_der_y-fd_y)<0.001)

    # TODO: 3D example. Not very important, because this is just multiplying, but still would be nice to do.

if __name__ == '__main__':
    unittest.main()