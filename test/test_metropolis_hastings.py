from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

from scipy.stats import multivariate_normal


class TestMetropolisHastings(unittest.TestCase):
    def test_analytic_1d(self):
        np.random.seed(0)
        # 1D test
        # Sample next point from a normal distribution
        def next_sample(x0):
            return np.array([multivariate_normal.rvs(x0,0.01)])

        # We want to sample a distribution that is proportional to this weird function
        # we don't know how to integrate and normalize
        def unnorm_distr(x):
            return np.max((1-np.abs(x[0]),1e-8))

        S, F = gpytoolbox.metropolis_hastings(unnorm_distr,next_sample,np.array([0.1]),1000000)
        # This should look like an absolute value pyramid function
        hist, bin_edges = np.histogram(S,bins=np.linspace(-1,1,101), density=True)
        bin_centers = (bin_edges[0:100] + bin_edges[1:101])/2.
        # Hard to know what a good value is here...
        self.assertTrue(np.mean(np.abs(hist - (1-np.abs(bin_centers))))<=0.03)
        # plot1 = plt.figure(1)
        # plt.hist(np.squeeze(S),100)
        # plt.title("Does this look like a pyramid with straight sides?")
        # plt.show(block=False)

    def test_analytic_2d(self):
        np.random.seed(0)
        # plt.pause(10)
        # plt.close(plot1)
        # 2D test
        # Next sample comes from a normal distribution
        def next_sample(x0):
            return multivariate_normal.rvs(x0,np.array([[0.01,0.0],[0.0,0.01]]))

        # We want to recover a normal function given a function proportional to its density
        def unnorm_distr(x):
            return 100*multivariate_normal.pdf(x,mean=np.array([0.0,0.0]),cov=np.array([[0.01,0.0],[0.0,0.01]]))

        S, F = gpytoolbox.metropolis_hastings(unnorm_distr,next_sample,np.array([0.01,0.01]),500000)

        nbins = 40
        H, xedges, yedges = np.histogram2d(S[:,0], S[:,1], density=True, bins=nbins)
        x_centers = (xedges[0:(nbins)] + xedges[1:(nbins+1)])/2.
        y_centers = (yedges[0:(nbins)] + yedges[1:(nbins+1)])/2.
        H = H.T
        # plt.imshow(H)
        # plt.show()

        H = np.reshape(H,(-1,1))
        x, y = np.meshgrid(np.linspace(x_centers[0],x_centers[nbins-1],nbins),np.linspace(y_centers[0],y_centers[nbins-1],nbins))
        verts = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        # print(H)
        # print(multivariate_normal.pdf(verts,mean=np.array([0.0,0.0]),cov=np.array([[0.01,0.0],[0.0,0.01]])))
        self.assertTrue(np.mean(np.abs(np.reshape(H,x.shape) - np.reshape(multivariate_normal.pdf(verts,mean=np.array([0.0,0.0]),cov=np.array([[0.01,0.0],[0.0,0.01]])),x.shape)[:]))<0.1)


if __name__ == '__main__':
    unittest.main()