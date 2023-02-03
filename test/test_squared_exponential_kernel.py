from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
# import matplotlib.pyplot as plt

class TestSquaredExponentialKernel(unittest.TestCase):
    def test_1d(self):
        np.random.seed(0)
        num_samples = 10000
        x = np.reshape(np.linspace(-1,1,num_samples),(-1,1))
        y = np.zeros(x.shape) + 0.2
        rand_center = 0.2
        # rand_scale = np.random.rand(1)
        v = gpytoolbox.squared_exponential_kernel(x,y)
        calculated_center = x[np.argmax(v)]
        # Center is the center
        self.assertTrue(np.isclose(rand_center-calculated_center,0.0,atol=1e-3))
        before_center = v[0:np.argmax(v)]
        after_center = v[np.argmax(v):num_samples]
        # Assert is monotonous
        self.assertTrue(np.all(before_center[:-1] <= before_center[1:]))
        self.assertTrue(np.all(after_center[:-1] >= after_center[1:]))
        # Now check derivatives:
        dx = x[1]-x[0]
        # First derivatives
        center = x[1:(num_samples-1)]
        fd = (v[2:num_samples] - v[0:(num_samples-2)] )/(2*dx)
        fun_der = gpytoolbox.squared_exponential_kernel(x[1:(num_samples-1)],y[1:(num_samples-1)],derivatives=(0,-1))
        self.assertTrue((np.abs(fun_der-fd)<0.001).all())
        # Second derivatives:
        fd = (-2*v[1:(num_samples-1)] + v[2:num_samples] + v[0:(num_samples-2)] )/(dx**2.0)
        fun_der = -gpytoolbox.squared_exponential_kernel(x[1:(num_samples-1)],y[1:(num_samples-1)],derivatives=(0,0))
        self.assertTrue((np.abs(fun_der-fd)<0.01).all())
    def test_2d(self):
        np.random.seed(0)
        num_samples = 2000
        x, y = np.meshgrid(np.linspace(-2,2,num_samples),np.linspace(-2,2,num_samples))
        V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        V2 = np.zeros(V.shape)
        h = x[1,1] - x[0,0]
        h = np.array([h,h])
        v = gpytoolbox.squared_exponential_kernel(V,V2)




        # CHECK FIRST DERIVATIVES 

        xs, ys = np.meshgrid(np.linspace(-2+0.5*h[0],2-0.5*h[0],num_samples-1),np.linspace(-2,2,num_samples))
        Vx = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wx = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,0)
        fd_x = Wx @ v
        fun_der_x = gpytoolbox.squared_exponential_kernel(Vx,np.zeros(Vx.shape),derivatives=(0,-1))
        self.assertTrue((np.abs(fun_der_x-fd_x)<0.001).all())



        xs, ys = np.meshgrid(np.linspace(-2,2,num_samples),np.linspace(-2+0.5*h[1],2-0.5*h[1],num_samples-1))
        Vy = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wy = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
        fd_y = Wy @ v
        fun_der_y = gpytoolbox.squared_exponential_kernel(Vy,np.zeros(Vy.shape),derivatives=(1,-1))
        self.assertTrue((np.abs(fun_der_y-fd_y)<0.001).all())



        # CHECK SECOND DERIVATIVES
        xs, ys = np.meshgrid(np.linspace(-2+h[0],2-h[0],num_samples-2),np.linspace(-2,2,num_samples))
        Vx = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wx = gpytoolbox.fd_partial_derivative(np.array([num_samples-1,num_samples]),h,0) @ gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,0)
        fd_x = Wx @ v
        # Note the minus sign!
        fun_der_x = -gpytoolbox.squared_exponential_kernel(Vx,np.zeros(Vx.shape),derivatives=(0,0))
        self.assertTrue((np.abs(fun_der_x-fd_x)<0.001).all())


        xs, ys = np.meshgrid(np.linspace(-2,2,num_samples),np.linspace(-2+h[0],2-h[0],num_samples-2))
        Vy = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wy = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples-1]),h,1) @ gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
        fd_y = Wy @ v
        # Note the minus sign!
        fun_der_y = -gpytoolbox.squared_exponential_kernel(Vy,np.zeros(Vy.shape),derivatives=(1,1))
        self.assertTrue((np.abs(fun_der_y-fd_y)<0.001).all())


        xs, ys = np.meshgrid(np.linspace(-2+0.5*h[0],2-0.5*h[0],num_samples-1),np.linspace(-2+0.5*h[1],2-0.5*h[1],num_samples-1))
        Vxy = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wxy = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples-1]),h,0) @ gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
        fd_xy = Wxy @ v
        # Note the minus sign!
        fun_der_xy = -gpytoolbox.squared_exponential_kernel(Vxy,np.zeros(Vxy.shape),derivatives=(1,0))
        self.assertTrue((np.abs(fun_der_xy-fd_xy)<0.001).all())

    def test_varying_length_scale(self):
        x1 = np.zeros(100)
        x2 = np.ones(100)
        lengths = np.linspace(1,1000,100)
        vals = gpytoolbox.squared_exponential_kernel(x1,x2,scale=1,length=lengths)
        # Must be positive
        self.assertTrue(np.all(vals>=0))
        # Must decrease monotonically
        # plt.plot(vals)
        # plt.show()
        self.assertTrue(np.all(vals[:-1] <= vals[1:]))


if __name__ == '__main__':
    unittest.main()

