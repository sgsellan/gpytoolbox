from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestSquaredDistanceToElement(unittest.TestCase):
    def test_random_point_pairs_2d(self):
        np.random.seed(0)
        for i in range(200):
            p = np.random.rand(1,2)
            V = np.random.rand(1,2)
            sqrD,_ = gpytoolbox.squared_distance_to_element(p,V,np.array([0]))
            distance_gt = np.linalg.norm(p-V)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - distance_gt,0.0,atol=1e-4))
    def test_random_point_pairs_3d(self):
        np.random.seed(0)
        for i in range(200):
            p = np.random.rand(1,3)
            V = np.random.rand(1,3)
            sqrD,_ = gpytoolbox.squared_distance_to_element(p,V,np.array([0]))
            distance_gt = np.linalg.norm(p-V)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - distance_gt,0.0,atol=1e-4))
    def test_random_edges_2d(self):
        #np.random.seed(0)
        V = np.array([[-1,0],[1,0]])
        edge = np.array([0,1],dtype=int)
        rndpts = 4*np.random.rand(100,2)-2
        th = 2*np.pi*np.random.rand(100)
        for i in range(rndpts.shape[0]):
            rndpt = rndpts[i,:]
            if rndpt[0]<-1:
                dist_gt = np.sum((np.array([-1,0]) - rndpt)**2.0)
            elif rndpt[0]>1:
                dist_gt = np.sum((np.array([1,0]) - rndpt)**2.0)
            else:
                dist_gt = rndpt[1]**2.0
            # Random rotation
            R = np.array([[np.cos(th[i]),np.sin(th[i])],[-np.sin(th[i]),np.cos(th[i])]])
            sqrD,_ = gpytoolbox.squared_distance_to_element(rndpt @ R.T,V @ R.T,edge)
            self.assertTrue(np.isclose(sqrD-dist_gt,0.0,atol=1e-5))
    def test_random_edges_3d(self):
        #np.random.seed(0)
        V = np.array([[-1,0,0],[1,0,0]])
        edge = np.array([0,1],dtype=int)
        rndpts = 4*np.random.rand(100,3)-2
        thx = 2*np.pi*np.random.rand(100)
        thy = 2*np.pi*np.random.rand(100)
        thz = 2*np.pi*np.random.rand(100)
        for i in range(rndpts.shape[0]):
            rndpt = rndpts[i,:]
            if rndpt[0]<-1:
                dist_gt = np.sum((np.array([-1,0,0]) - rndpt)**2.0)
            elif rndpt[0]>1:
                dist_gt = np.sum((np.array([1,0,0]) - rndpt)**2.0)
            else:
                dist_gt = np.sum(rndpt[1:3]**2.0)
            # Random rotation
            Rz = np.array([[np.cos(thx[i]),np.sin(thx[i]),0],[-np.sin(thx[i]),np.cos(thx[i]),0],[0,0,1]])
            Ry = np.array([[ np.cos(thy[i]),0,np.sin(thy[i]) ],[0,1,0], [ -np.sin(thy[i]),0,np.cos(thy[i]) ]])
            Rx = np.array([[1,0,0],[0,np.cos(thx[i]),np.sin(thx[i])],[0,-np.sin(thx[i]),np.cos(thx[i])]])
            sqrD,_ = gpytoolbox.squared_distance_to_element(rndpt @ Rz.T @ Ry.T @ Rx.T,V @ Rz.T @ Ry.T @ Rx.T,edge)
            self.assertTrue(np.isclose(sqrD-dist_gt,0.0,atol=1e-5))


        




if __name__ == '__main__':
    unittest.main()
