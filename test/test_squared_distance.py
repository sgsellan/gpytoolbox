from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestSquaredDistance(unittest.TestCase):
    def is_consistent(self,sqrD1,sqrD2,ind1,ind2,lmbd1,lmbd2,V,F):
        dim = V.shape[1]
        ss = F.shape[1]
        n = sqrD1.shape[0]

        # Check that the distances are the same
        self.assertTrue(np.isclose(sqrD1-sqrD2,0).all())
        
        # Check that the closest points are the same
        closest_points1 = np.zeros((n,dim))
        closest_points2 = np.zeros((n,dim))
        for i in range(dim):
            for j in range(ss):
                closest_points1[:,i] += lmbd1[:,j]*V[F[ind1,j],i]
                closest_points2[:,i] += lmbd2[:,j]*V[F[ind2,j],i]
        # print(closest_points1)
        # print(closest_points2)
        self.assertTrue(np.isclose(closest_points1-closest_points2,0,atol=1e-5).all())




        

    def test_find_closest_point_2d_pointcloud(self):
        np.random.seed(0)
        ss = 20
        for ss in range(10,2000,100):       
            P = np.random.rand(ss,2)
            ptest = P[9,:] + 1e-5
            sqrD,ind,lmb = gpytoolbox.squared_distance(ptest,P,use_aabb=False)
            sqrD2,ind2,lmb2 = gpytoolbox.squared_distance(ptest,P,use_cpp=True)
            # print(np.sqrt(sqrD))
            # print(sqrD2)
            # print(ind2)
            # print(lmb2)
            self.assertTrue(ind2==9)
            self.assertTrue(ind==9)
            self.assertTrue(lmb==1)
            self.assertTrue(lmb2==1)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(2)*1e-5,0,atol=1e-5))
            self.assertTrue(np.isclose(np.sqrt(sqrD2) - np.sqrt(2)*1e-5,0,atol=1e-5))
            sqrD,ind,lmb = gpytoolbox.squared_distance(ptest,P,use_aabb=True)
            self.assertTrue(ind==9)
            self.assertTrue(lmb==1)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(2)*1e-5,0,atol=1e-5))
    def test_find_closest_point_3d_pointcloud(self):
        np.random.seed(0)
        ss = 20
        for ss in range(10,2000,100):       
            P = np.random.rand(ss,3)
            ptest = P[9,:] + 1e-5
            sqrD,ind,lmb = gpytoolbox.squared_distance(ptest,P,use_aabb=False)
            # print(np.sqrt(sqrD))
            self.assertTrue(ind==9)
            self.assertTrue(lmb==1)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(3)*1e-5,0,atol=1e-5))
            sqrD,ind,lmb = gpytoolbox.squared_distance(ptest,P,use_aabb=True)
            # print(np.sqrt(sqrD))
            self.assertTrue(ind==9)
            self.assertTrue(lmb==1)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(3)*1e-5,0,atol=1e-5))
    def test_polygon_synthetic(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])
        sample_points = np.array([  [0.01,0.0],
                                    [0.3,0.0],
                                    [-1.5,0.5],
                                    [1.2,0.0]])
        groundtruth_vals = np.array([0.99,0.7,0.5,0.2])**2.0
        E = gpytoolbox.edge_indices(V.shape[0])
        for i in range(sample_points.shape[0]):
            sqrD,ind,lmb = gpytoolbox.squared_distance(sample_points[i,:],V,F=E,use_aabb=False)
            # print(groundtruth_vals[i])
            self.assertTrue(np.isclose(sqrD-groundtruth_vals[i],0).all())
            sqrD,ind,lmb = gpytoolbox.squared_distance(sample_points[i,:],V,F=E,use_aabb=True)
            # print(sqrD)
            # print(groundtruth_vals[i])
            self.assertTrue(np.isclose(sqrD-groundtruth_vals[i],0).all())
        # All together
        sqrD_gt,ind_gt,lmb_gt = gpytoolbox.squared_distance(sample_points,V,F=E,use_aabb=False)
        # self.assertTrue(np.isclose(sqrsD-groundtruth_vals,0).all())
        sqrD_aabb,ind_aabb,lmb_aabb = gpytoolbox.squared_distance(sample_points,V,F=E,use_aabb=True)
        sqrD_cpp,ind_cpp,lmb_cpp = gpytoolbox.squared_distance(sample_points,V,F=E,use_cpp=True)
        self.is_consistent(sqrD_aabb,sqrD_gt,ind_aabb,ind_gt,lmb_aabb,lmb_gt,V,E)
        self.is_consistent(sqrD_gt,sqrD_cpp,ind_gt,ind_cpp,lmb_gt,lmb_cpp,V,E)
    def test_polygon_from_image(self):
        filename = "test/unit_tests_data/poly.png"
        poly = gpytoolbox.png2poly(filename)
        V = gpytoolbox.normalize_points(poly[0])
        V = V[0:V.shape[0]:100,:]
        # print(V.shape[0])
        E = gpytoolbox.edge_indices(V.shape[0])
        P = 2*np.random.rand(100,2)-4
        for i in range(10):
            sqrD_gt,ind,lmb = gpytoolbox.squared_distance(P[i,:],V,F=E,use_aabb=False)
            # print(groundtruth_vals[i])
            sqrD_aabb,ind,lmb = gpytoolbox.squared_distance(P[i,:],V,F=E,use_aabb=True)
            self.assertTrue(np.isclose(sqrD_aabb-sqrD_gt,0).all())
        # All together now
        sqrD_gt,ind_gt,lmb_gt = gpytoolbox.squared_distance(P,V,F=E,use_aabb=False)
            # print(groundtruth_vals[i])
        sqrD_aabb,ind_aabb,lmb_aabb = gpytoolbox.squared_distance(P,V,F=E,use_aabb=True)
        sqrD_cpp, ind_cpp, lmb_cpp = gpytoolbox.squared_distance(P,V,F=E,use_cpp=True)
        self.is_consistent(sqrD_aabb,sqrD_gt,ind_aabb,ind_gt,lmb_aabb,lmb_gt,V,E)
        self.is_consistent(sqrD_gt,sqrD_cpp,ind_gt,ind_cpp,lmb_gt,lmb_cpp,V,E)


    def test_polygon_from_image_3d(self):
        np.random.seed(0)
        filename = "test/unit_tests_data/illustrator.png"
        poly = gpytoolbox.png2poly(filename)
        V = gpytoolbox.normalize_points(poly[0])
        V = V[0:V.shape[0]:10,:]
        V = np.hstack(( V,np.zeros((V.shape[0],1)) ))
        num_samples = 100
        thx = 2*np.pi*np.random.rand(num_samples)
        thy = 2*np.pi*np.random.rand(num_samples)
        thz = 2*np.pi*np.random.rand(num_samples)
        # print(V.shape[0])
        E = gpytoolbox.edge_indices(V.shape[0])
        P = 2*np.random.rand(num_samples,3)-4
        for i in range(P.shape[0]):
            Rz = np.array([[np.cos(thx[i]),np.sin(thx[i]),0],[-np.sin(thx[i]),np.cos(thx[i]),0],[0,0,1]])
            Ry = np.array([[ np.cos(thy[i]),0,np.sin(thy[i]) ],[0,1,0], [ -np.sin(thy[i]),0,np.cos(thy[i]) ]])
            Rx = np.array([[1,0,0],[0,np.cos(thz[i]),np.sin(thz[i])],[0,-np.sin(thz[i]),np.cos(thz[i])]])
            V = V @ Rx.T @ Ry.T @ Rz.T
            sqrD_gt,ind,lmb = gpytoolbox.squared_distance(P[i,:],V,F=E,use_aabb=False)
            # print(ind)
            # print(groundtruth_vals[i])
            sqrD_aabb,ind,lmb = gpytoolbox.squared_distance(P[i,:],V,F=E,use_aabb=True)
            self.assertTrue(np.isclose(sqrD_aabb-sqrD_gt,0).all())
        # All together now
        sqrD_gt,ind_gt,lmb_gt = gpytoolbox.squared_distance(P,V,F=E,use_aabb=False)
        sqrD_aabb,ind_aabb,lmb_aabb = gpytoolbox.squared_distance(P,V,F=E,use_aabb=True)
        sqrD_cpp,ind_cpp,lmb_cpp = gpytoolbox.squared_distance(P,V,F=E,use_cpp=True)
        self.is_consistent(sqrD_aabb,sqrD_gt,ind_aabb,ind_gt,lmb_aabb,lmb_gt,V,E)
        self.is_consistent(sqrD_gt,sqrD_cpp,ind_gt,ind_cpp,lmb_gt,lmb_cpp,V,E)


    def test_meshes(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "bunny.obj", "mountain.obj"]
        num_samples = 10 # Should be more but this is already pretty slow
        for mesh in meshes:
            v,f = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
            v = gpytoolbox.normalize_points(v)
            v,f,_,_ = gpytoolbox.decimate(v,f,face_ratio=0.1)
            # print(f.shape[0])
            # Generate random point
            P = 2*np.random.rand(num_samples,3)-4
            for i in range(P.shape[0]):
                # print(i)
                sqrD_gt,ind,lmb = gpytoolbox.squared_distance(P[i,:],v,F=f,use_aabb=False)
    #         # print(groundtruth_vals[i])
                sqrD_aabb,ind,lmb = gpytoolbox.squared_distance(P[i,:],v,F=f,use_aabb=True)
                self.assertTrue(np.isclose(sqrD_aabb-sqrD_gt,0).all())
            sqrD_gt,ind_gt,lmb_gt = gpytoolbox.squared_distance(P,v,F=f,use_aabb=False)
            sqrD_aabb,ind_aabb,lmb_aabb = gpytoolbox.squared_distance(P,v,F=f,use_aabb=True)
            sqrD_cpp,ind_cpp,lmb_cpp = gpytoolbox.squared_distance(P,v,F=f,use_cpp=True)
    #         # print(groundtruth_vals[i])

            # Use precomputed tree
            C,W,CH,_,_,tri_ind = gpytoolbox.initialize_aabbtree(v,F=f)
            sqrD_aabb_2,ind_aabb_2,lmb_aabb_2 = gpytoolbox.squared_distance(P,v,F=f,use_aabb=True,C=C,W=W,tri_ind=tri_ind,CH=CH)
            self.is_consistent(sqrD_aabb,sqrD_gt,ind_aabb,ind_gt,lmb_aabb,lmb_gt,v,f)
            self.is_consistent(sqrD_aabb_2,sqrD_gt,ind_aabb_2,ind_gt,lmb_aabb_2,lmb_gt,v,f)
            self.is_consistent(sqrD_gt,sqrD_cpp,ind_gt,ind_cpp,lmb_gt,lmb_cpp,v,f)





if __name__ == '__main__':
    unittest.main()
