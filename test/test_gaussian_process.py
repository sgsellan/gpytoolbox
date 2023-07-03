from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import matplotlib.pyplot as plt

class TestGaussianProcess(unittest.TestCase):
    def test_straight_line(self):
        def true_fun(x):
            return 2*x

        # Test something
        x_train = np.linspace(0,1,20)
        y_train = true_fun(x_train)


        # gp.train(np.reshape(x_train,(-1,1)),y_train)

        x_test = np.linspace(0,1,120)
        y_test_mean,y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)))
        # print(np.asarray(y_test_cov).diagonal().shape)
        # print(np.max(np.abs(y_test_mean - true_fun(x_test))))
        self.assertTrue(np.isclose(y_test_mean - true_fun(x_test),0,atol=0.01).all())

        # compact_kernel = gpytoolbox.compactly_supported_normal_kernel()
        def kernel(X1,X2):
            return gpytoolbox.compactly_supported_normal_kernel(X1,X2,length=0.5)
        y_test_mean, y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)),kernel=kernel)
        self.assertTrue(np.isclose(y_test_mean - true_fun(x_test),0,atol=0.01).all())
        y_test_mean, y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)),kernel=kernel,compact_kernel=True)
        self.assertTrue(np.isclose(y_test_mean - true_fun(x_test),0,atol=0.01).all())

        # print(y_test_cov.diagonal().flatten())
        # plt.scatter(x_test,y_test_mean,c=np.asarray(y_test_cov).diagonal())
        # plt.show()
    def test_inducing_1d(self):
        # This tests that trivially, the if the induced and training points are the same, the gp is the same

        def true_fun(x):
            return np.cos(10*x)
        x_train = np.linspace(0,1,30)
        y_train = true_fun(x_train)

        # gp_kernel = utility.gp_kernel(dim=1,type=k,scale=1.0)
        # gp = utility.gaussian_process(gp_kernel,verbose=False)
        # gp.train(np.reshape(x_train,(-1,1)),y_train)
        # gp_induced = utility.gaussian_process(gp_kernel,verbose=False)
        # gp_induced.train(np.reshape(x_train,(-1,1)),y_train,X_induced=np.reshape(x_train,(-1,1)))

        x_test = np.linspace(0,1,120)
        y_test_mean,y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)))
        y_test_mean_ind,y_test_cov_ind = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)),X_induced=np.reshape(x_train,(-1,1)))
        # print(y_test_mean-y_test_mean_ind)
        # print(y_test_cov-y_test_cov_ind)
        self.assertTrue(np.isclose(y_test_mean_ind - y_test_mean,0,atol=0.1).all())
        
        self.assertTrue(np.isclose(y_test_cov_ind - y_test_cov,0,atol=0.1).all())
    

    # TODO: FIGURE THIS OUT: 
    def test_inducing_2d(self):

        gs = 60
        gx, gy = np.meshgrid(np.linspace(-1,1,gs),np.linspace(-1,1,gs))
        x_test = np.concatenate((np.reshape(gx,(-1, 1)),np.reshape(gy,(-1, 1))),axis=1)

        P = np.array([[-0.44898559, -0.032634  ],
            [ 0.2024951,   0.53782643],
            [-0.51685489,  0.17814241],
            [-0.05698656, -0.72474838],
            [ 0.18569608, -0.03954301],
            [-0.44956696, -0.03350001],
            [-0.63302571,  0.69562693],
            [ 0.26923987, -0.72325672],
            [-0.79638273,  0.44638672],
            [-0.45209117, -0.03993589],
            [ 0.0319679 ,  0.39117874],
            [-0.00577538, -0.72823037],
            [-0.12220172 , 0.41366594],
            [ 0.17600946 ,-0.06311775],
            [-0.21941915 ,-0.70611408],
            [-0.36828304, -0.65738441],
            [ 0.53678448, -0.18512404],
            [-0.44385065, -0.00865092],
            [ 0.31854042,  0.57413826],
            [-0.4625986 ,  0.13619191],
            [ 0.7383548 , -0.22352159],
            [-0.73453125,  0.60272283],
            [ 0.21145535,  0.5450711 ],
            [ 0.65464397, -0.60493385],
            [-0.66664002, -0.4095675 ],
            [ 0.20076894, -0.21482662],
            [-0.57421545,  0.19430317],
            [-0.46734197, -0.07878698],
            [-0.48861026, -0.1123013 ],
            [ 0.25519813, -0.20736916],
            [ 0.64317788, -0.61438431],
            [ 0.41321472, -0.19025555],
            [ 0.16388205,  0.50151875],
            [-0.04327652 ,-0.72475297],
            [-0.21296902, -0.70731427],
            [ 0.22162023,  0.01507432],
            [ 0.16940526, -0.08082856],
            [-0.24578988, -0.69980738],
            [ 0.7548747 , -0.23655855],
            [ 0.21529399 , 0.5480239 ]])
        # print(P)
        inside = np.array([[-0.0,0.0]])

        th = np.reshape(np.linspace(0,2*np.pi,20),(-1,1))
        outside = 1.3*np.concatenate((np.cos(th),np.sin(th)),axis=1)

        # outside = np.array([[-1,-1],[1,1],[-1,1],[1,-1]])
        X_train = np.vstack((
            P,
            inside,
            outside
        ))
        y_train = np.concatenate((
            np.zeros(P.shape[0]),np.ones(inside.shape[0]),-np.ones(outside.shape[0])
        ))
        # gp_kernel = utility.gp_kernel(dim=2,type='exponential',length=0.1,scale=0.1)
        # gp_gt = utility.gaussian_process(gp_kernel,verbose=False)
        # gp_gt.train(X_train,y_train)
        # mean,cov = gp_gt(x_test)
        # gp_induced_gt = utility.gaussian_process(gp_kernel,verbose=False)
        # gp_induced_gt.train(X_train,y_train,X_induced=X_train)
        # mean_ind,cov_ind = gp_induced_gt(x_test)
        def ker_fun(X1,X2,derivatives=(-1,-1)):
            return gpytoolbox.squared_exponential_kernel(X1,X2,derivatives=derivatives,length=0.4,scale=0.1)
        mean,cov = gpytoolbox.gaussian_process(X_train,y_train,x_test,kernel=ker_fun)
        # plt.imshow(np.reshape(mean,(-1,gs)))
        # plt.show()
            
    def test_with_gradients(self):
        def true_fun(x):
            return 10*x
        def true_grad(x):
            return 10 + 0*x

        # Test something
        x_train = np.linspace(0,1,20)
        y_train = true_fun(x_train)
        y_grad = true_grad(x_train)
        # gp.train(np.reshape(x_train,(-1,1)),y_train)

        x_test = np.linspace(0,1,140)
        y_test_mean,y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)),grad_y_train=y_grad)
        # print(y_test_cov)
        y_test_mean_plot = y_test_mean[0:len(x_test)]
        y_test_cov_plot = np.asarray(y_test_cov).diagonal()[0:len(x_test)]
        # plt.plot(x_test,true_fun(x_test))
        # plt.plot(x_test,y_test_mean_plot)
        # plt.show()
        self.assertTrue(np.isclose(y_test_mean_plot - true_fun(x_test),0,atol=0.01).all())
    def test_precomputation(self):
        def true_fun(x):
            return 10*x
        def true_grad(x):
            return 10 + 0*x

        # Test something
        x_train = np.linspace(0,1,20)
        y_train = true_fun(x_train)
        y_grad = true_grad(x_train)
        # gp.train(np.reshape(x_train,(-1,1)),y_train)

        x_test = np.linspace(0,1,140)
        y_test_mean,y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)),grad_y_train=y_grad)
        y_test_mean_2, y_test_cov_2 = gpytoolbox.gaussian_process_precompute(np.reshape(x_train,(-1,1)),y_train,grad_y_train=y_grad).predict(np.reshape(x_test,(-1,1)))
        self.assertTrue(np.isclose(y_test_mean_2 - y_test_mean,0,atol=0.01).all())
        self.assertTrue(np.isclose(y_test_cov - y_test_cov_2,0,atol=0.01).all())

    def test_verbose(self):
        def true_fun(x):
            return 10*x
        def true_grad(x):
            return 10 + 0*x

        # Test something
        x_train = np.linspace(0,1,20)
        y_train = true_fun(x_train)
        y_grad = true_grad(x_train)
        # gp.train(np.reshape(x_train,(-1,1)),y_train)

        x_test = np.linspace(0,1,140)
        y_test_mean,y_test_cov = gpytoolbox.gaussian_process(np.reshape(x_train,(-1,1)),y_train,np.reshape(x_test,(-1,1)),grad_y_train=y_grad,verbose=True)

if __name__ == '__main__':
    unittest.main()

