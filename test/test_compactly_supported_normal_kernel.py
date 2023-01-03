from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import matplotlib.pyplot as plt


class TestCompactlySupportedNormalKernel(unittest.TestCase):
    def test_2d(self):
        np.random.seed(0)
        num_samples = 403
        x, y = np.meshgrid(np.linspace(-2,2,num_samples),np.linspace(-2,2,num_samples))
        V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        V2 = np.zeros(V.shape)
        h = x[1,1] - x[0,0]
        h = np.array([h,h])
        v = gpytoolbox.compactly_supported_normal_kernel(V,V2,scale=0.5,length=0.5)
        # print(v)



        # CHECK FIRST DERIVATIVES 

        xs, ys = np.meshgrid(np.linspace(-2+0.5*h[0],2-0.5*h[0],num_samples-1),np.linspace(-2,2,num_samples))
        Vx = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wx = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,0)
        fd_x = Wx @ v
        fun_der_x = gpytoolbox.compactly_supported_normal_kernel(Vx,np.zeros(Vx.shape),scale=0.5,derivatives=(0,-1),length=0.5)
        # plt.figure()
        # plt.imshow(np.reshape(fun_der_x,(num_samples-1,num_samples),order='F'))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.reshape(fd_x,(num_samples-1,num_samples),order='F'))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.reshape(fd_x-fun_der_x,(num_samples-1,num_samples),order='F'))
        # plt.colorbar()
        # plt.show()
        self.assertTrue((np.abs(fun_der_x-fd_x)<0.001).all())



        xs, ys = np.meshgrid(np.linspace(-2,2,num_samples),np.linspace(-2+0.5*h[1],2-0.5*h[1],num_samples-1))
        Vy = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wy = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
        fd_y = Wy @ v
        fun_der_y = gpytoolbox.compactly_supported_normal_kernel(Vy,np.zeros(Vy.shape),scale=0.5,derivatives=(1,-1),length=0.5)
        self.assertTrue((np.abs(fun_der_y-fd_y)<0.001).all())



        # CHECK SECOND DERIVATIVES
        xs, ys = np.meshgrid(np.linspace(-2+h[0],2-h[0],num_samples-2),np.linspace(-2,2,num_samples))
        Vx = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wx = gpytoolbox.fd_partial_derivative(np.array([num_samples-1,num_samples]),h,0) @ gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,0)
        fd_x = Wx @ v
        # Note the minus sign!
        fun_der_x = -gpytoolbox.compactly_supported_normal_kernel(Vx,np.zeros(Vx.shape),scale=0.5,derivatives=(0,0),length=0.5)
        
        # self.assertTrue((np.abs(fun_der_x-fd_x)<0.01).all())
        ind = np.argmax(np.abs(fun_der_x-fd_x))
        print(np.abs(fun_der_x-fd_x)[ind])
        print(fun_der_x[ind])
        print(fd_x[ind])
        print(Vx[ind,:])
        plt.figure()
        plt.imshow(np.reshape(fun_der_x,(num_samples-2,num_samples),order='F'))
        plt.colorbar()
        plt.figure()
        plt.imshow(np.reshape(fd_x,(num_samples-2,num_samples),order='F'))
        plt.colorbar()
        plt.figure()
        plt.imshow(np.reshape(fd_x-fun_der_x,(num_samples-2,num_samples),order='F'))
        plt.colorbar()
        plt.show()
        self.assertTrue((np.abs(fun_der_x-fd_x)<0.01).all())
        # self.assertTrue(np.median(np.abs(fun_der_x-fd_x))<0.001)


        xs, ys = np.meshgrid(np.linspace(-2,2,num_samples),np.linspace(-2+h[0],2-h[0],num_samples-2))
        Vy = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wy = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples-1]),h,1) @ gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
        fd_y = Wy @ v
        # Note the minus sign!
        fun_der_y = -gpytoolbox.compactly_supported_normal_kernel(Vy,np.zeros(Vy.shape),scale=0.5,derivatives=(1,1),length=0.5)
        self.assertTrue((np.abs(fun_der_y-fd_y)<0.01).all())
        # self.assertTrue(np.median(np.abs(fun_der_y-fd_y))<0.001)

        xs, ys = np.meshgrid(np.linspace(-2+0.5*h[0],2-0.5*h[0],num_samples-1),np.linspace(-2+0.5*h[1],2-0.5*h[1],num_samples-1))
        Vxy = np.concatenate((np.reshape(xs,(-1, 1)),np.reshape(ys,(-1, 1))),axis=1)
        Wxy = gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples-1]),h,0) @ gpytoolbox.fd_partial_derivative(np.array([num_samples,num_samples]),h,1)
        fd_xy = Wxy @ v
        # Note the minus sign!
        fun_der_xy = -gpytoolbox.compactly_supported_normal_kernel(Vxy,np.zeros(Vxy.shape),scale=0.5,derivatives=(1,0),length=0.5)
        ind = np.argmax(np.abs(fun_der_xy-fd_xy))
        print(np.abs(fun_der_xy-fd_xy)[ind])
        print(fun_der_xy[ind])
        print(fd_xy[ind])
        print(Vxy[ind,:])
        # plt.figure()
        # plt.imshow(np.reshape(fun_der_xy,(num_samples-1,num_samples-1),order='F'))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.reshape(fd_xy,(num_samples-1,num_samples-1),order='F'))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.reshape(fd_xy-fun_der_xy,(num_samples-1,num_samples-1),order='F'))
        # plt.colorbar()
        # plt.show()
        self.assertTrue((np.abs(fun_der_xy-fd_xy)<0.01).all())
        # self.assertTrue(np.median(np.abs(fun_der_xy-fd_xy))<0.001)


if __name__ == '__main__':
    unittest.main()

