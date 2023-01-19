import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestSignedDistance(unittest.TestCase):
    # This isn't too complex, probably could use being expanded
    def test_synthetic(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])
        sample_points = np.array([  [0.0,0.0],
                                    [0.3,0.0],
                                    [-1.5,0.5],
                                    [1.2,0.0]])
        groundtruth_vals = np.array([-1.0,-0.7,0.5,0.2])
        EC = gpytoolbox.edge_indices(V.shape[0])
        S = gpytoolbox.signed_distance(sample_points,V,EC)[0]
        self.assertTrue(np.isclose(S-groundtruth_vals,0).all())
        S = gpytoolbox.signed_distance(sample_points,V)[0]
        self.assertTrue(np.isclose(S-groundtruth_vals,0).all())
    def test_duplicated(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0],[-1.0, -1.0] ])
        sample_points = np.array([  [0.0,0.0],
                                    [0.3,0.0],
                                    [-1.5,0.5],
                                    [1.2,0.0]])
        groundtruth_vals = np.array([-1.0,-0.7,0.5,0.2])
        EC = gpytoolbox.edge_indices(V.shape[0])
        S = gpytoolbox.signed_distance(sample_points,V,EC)[0]
        self.assertTrue(np.isclose(S-groundtruth_vals,0).all())
        S = gpytoolbox.signed_distance(sample_points,V)[0]
        self.assertTrue(np.isclose(S-groundtruth_vals,0).all())

    def test_meshes_magnitude(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "bunny.obj", "mountain.obj"]
        num_samples = 10 # Should be more but this is already pretty slow
        for mesh in meshes:
            v,f = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
            v = gpytoolbox.normalize_points(v)
            v,f,_,_ = gpytoolbox.decimate(v,f,face_ratio=0.1)
            # print(f.shape[0])
            # Generate random point
            P = 2*np.random.rand(num_samples,3)-4
            sqrD_gt,ind_gt,lmb_gt = gpytoolbox.squared_distance(P,v,F=f,use_cpp=True)
            dist_1,ind_1,lmb_1 = gpytoolbox.signed_distance(P,v,f,use_cpp=True)
            dist_2,ind_2,lmb_2 = gpytoolbox.signed_distance(P,v,f,use_cpp=False)
            self.is_consistent(sqrD_gt,dist_1**2.0,ind_gt,ind_1,lmb_gt,lmb_1,v,f)
            self.is_consistent(sqrD_gt,dist_2**2.0,ind_gt,ind_2,lmb_gt,lmb_2,v,f)
    def test_sign_bunny(self):
        meshes = ["bunny_oded.obj", "cube.obj"]
        bools = [True, False]
        for mesh in meshes:
            for bbool in bools:
                num_points = 1000
                V,F = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
                # Generate random points on mesh
                Q,I,_ = gpytoolbox.random_points_on_mesh(V,F,num_points,return_indices=True,rng=np.random.default_rng(5))
                # Per face normals
                N = gpytoolbox.per_face_normals(V,F,unit_norm=True)
                # Compute winding number
                eps = 1e-3
                points_out = Q + eps*N[I,:]
                points_in = Q - eps*N[I,:]
                s_in = np.sign(gpytoolbox.signed_distance(points_in,V,F,use_cpp=bbool)[0])
                s_out = np.sign(gpytoolbox.signed_distance(points_out,V,F,use_cpp=bbool)[0])
                # print(wn_in)
                # print(wn_out)
                # print(np.isclose(wn_out,0,atol=1e-2))
                self.assertTrue(np.isclose(s_out,-1,atol=1e-2).all())
                self.assertTrue(np.isclose(s_in,1,atol=1e-2).all())



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


        

if __name__ == '__main__':
    unittest.main()
