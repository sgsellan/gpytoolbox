from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestSquaredDistanceToElement(unittest.TestCase):
    def test_random_point_pairs_2d(self):
        np.random.seed(0)
        for i in range(200):
            p = np.random.rand(1,2)
            V = np.random.rand(1,2)
            sqrD,lmb = gpytoolbox.squared_distance_to_element(p,V,np.array([0]))
            distance_gt = np.linalg.norm(p-V)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - distance_gt,0.0,atol=1e-4))
            self.assertTrue(lmb==1)
    def test_random_point_pairs_3d(self):
        np.random.seed(0)
        for i in range(200):
            p = np.random.rand(1,3)
            V = np.random.rand(1,3)
            sqrD,lmb = gpytoolbox.squared_distance_to_element(p,V,np.array([0]))
            distance_gt = np.linalg.norm(p-V)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - distance_gt,0.0,atol=1e-4))
            self.assertTrue(lmb==1)
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
                nearest_point = np.array([-1,0])
            elif rndpt[0]>1:
                dist_gt = np.sum((np.array([1,0]) - rndpt)**2.0)
                nearest_point = np.array([1,0])
            else:
                dist_gt = rndpt[1]**2.0
                nearest_point = np.array([rndpt[0],0])
            # Random rotation
            R = np.array([[np.cos(th[i]),np.sin(th[i])],[-np.sin(th[i]),np.cos(th[i])]])
            sqrD,lmb = gpytoolbox.squared_distance_to_element(rndpt @ R.T,V @ R.T,edge)
            self.assertTrue(np.isclose(sqrD-dist_gt,0.0,atol=1e-5))
            # print(lmb[0]*V[0,:] + lmb[1]*V[1,:])
            # print(nearest_point)
            self.assertTrue(np.isclose( lmb[0]*V[0,:] + lmb[1]*V[1,:] - nearest_point,0).all())
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
                nearest_point = np.array([-1,0,0])
            elif rndpt[0]>1:
                dist_gt = np.sum((np.array([1,0,0]) - rndpt)**2.0)
                nearest_point = np.array([1,0,0])
            else:
                dist_gt = np.sum(rndpt[1:3]**2.0)
                nearest_point = np.array([rndpt[0],0,0])
            # Random rotation
            Rz = np.array([[np.cos(thx[i]),np.sin(thx[i]),0],[-np.sin(thx[i]),np.cos(thx[i]),0],[0,0,1]])
            Ry = np.array([[ np.cos(thy[i]),0,np.sin(thy[i]) ],[0,1,0], [ -np.sin(thy[i]),0,np.cos(thy[i]) ]])
            Rx = np.array([[1,0,0],[0,np.cos(thz[i]),np.sin(thz[i])],[0,-np.sin(thz[i]),np.cos(thz[i])]])
            sqrD,lmb = gpytoolbox.squared_distance_to_element(rndpt @ Rz.T @ Ry.T @ Rx.T,V @ Rz.T @ Ry.T @ Rx.T,edge)
            self.assertTrue(np.isclose(sqrD-dist_gt,0.0,atol=1e-5))
            self.assertTrue(np.isclose( lmb[0]*V[0,:] + lmb[1]*V[1,:] - nearest_point,0).all())

    def test_random_triangle(self):
        # Generate random triangle
        np.random.seed(0)
        num_tests = 100
        for i in range(num_tests):
            V = np.random.rand(3,3)
            F = np.array([0,1,2],dtype=int)
            # Generate random query point
            P = np.random.rand(3)
            # Calculate distance with our method
            sqrD,lmb = gpytoolbox.squared_distance_to_element(P,V,F)
            # Now, generate many random points on the triangle
            num_samples = 10000000
            s = np.random.rand(num_samples,1)
            t = np.random.rand(num_samples,1)
            #b = np.array([1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)])
            b = np.hstack( (1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)) )
            rand_points = np.vstack( (
                b[:,0]*V[0,0] + b[:,1]*V[1,0] + b[:,2]*V[2,0],
                b[:,0]*V[0,1] + b[:,1]*V[1,1] + b[:,2]*V[2,1],
                b[:,0]*V[0,2] + b[:,1]*V[1,2] + b[:,2]*V[2,2]
            )).T
            smallest_rand_distance = np.amin( np.sum((np.tile(P[None,:],(num_samples,1)) - rand_points)**2.0,axis=1) )
            best_rand_guess = np.argmin( np.sum((np.tile(P[None,:],(num_samples,1)) - rand_points)**2.0,axis=1) )
            # Is our computed distance close to the minimum distance to the random points
            self.assertTrue(np.isclose(sqrD-smallest_rand_distance,0.0,atol=1e-3))
            self.assertTrue(np.isclose(b[best_rand_guess,:]-lmb,0.0,atol=1e-2).all())




        




if __name__ == '__main__':
    unittest.main()
