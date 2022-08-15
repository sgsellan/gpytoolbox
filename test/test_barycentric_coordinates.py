from .context import numpy as np
from .context import unittest
from .context import gpytoolbox

class TestBarycentricCoordinates(unittest.TestCase):
    def test_random_3d_triangles(self):
        for j in range(100):
            # Generate random triangle
            num_samples = 100
            v1 = np.random.rand(3)
            v2 = np.random.rand(3)
            v3 = np.random.rand(3)
            s = np.random.rand(num_samples,1)
            t = np.random.rand(num_samples,1)
            #b = np.array([1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)])
            b = np.hstack( (1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)) )
            rand_points = np.vstack( (
                    b[:,0]*v1[0] + b[:,1]*v2[0] + b[:,2]*v3[0],
                    b[:,0]*v1[1] + b[:,1]*v2[1] + b[:,2]*v3[1],
                    b[:,0]*v1[2] + b[:,1]*v2[2] + b[:,2]*v3[2]
                )).T
            for i in range(rand_points.shape[0]):
                b_ours = gpytoolbox.barycentric_coordinates(rand_points[i,:],v1,v2,v3)
                self.assertTrue(np.isclose(b[i,:]-b_ours,0).all())
                # All should be positive since we're in the triangle
                self.assertTrue((b_ours>=0).all())
            # Now test out of triangle: Generate random points in the triangle plane
            u = 5*np.random.rand(num_samples,2)-10
            v12 = v2 - v1
            v13 = v3 - v1
            rand_points = np.vstack( (
                    v1[0] + u[:,0]*v12[0] + u[:,1]*v13[0],
                    v1[1] + u[:,0]*v12[1] + u[:,1]*v13[1],
                    v1[2] + u[:,0]*v12[2] + u[:,1]*v13[2]
                )).T
            # Theses are not unique so require more care
            b_ours_big = np.zeros((num_samples,3))
            for i in range(rand_points.shape[0]):
                b_ours_big[i,:] = gpytoolbox.barycentric_coordinates(rand_points[i,:],v1,v2,v3)
                # print(b_ours_big[i,:])
            our_points = np.vstack( (
                    b_ours_big[:,0]*v1[0] + b_ours_big[:,1]*v2[0] + b_ours_big[:,2]*v3[0],
                    b_ours_big[:,0]*v1[1] + b_ours_big[:,1]*v2[1] + b_ours_big[:,2]*v3[1],
                    b_ours_big[:,0]*v1[2] + b_ours_big[:,1]*v2[2] + b_ours_big[:,2]*v3[2]
                )).T
            # print(our_points)
            # print(rand_points)
            self.assertTrue(np.isclose(our_points-rand_points,0.0).all())
    def test_random_2d_triangles(self):
        for j in range(100):
            # Generate random triangle
            num_samples = 100
            v1 = np.random.rand(2)
            v2 = np.random.rand(2)
            v3 = np.random.rand(2)
            s = np.random.rand(num_samples,1)
            t = np.random.rand(num_samples,1)
            #b = np.array([1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)])
            b = np.hstack( (1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)) )
            rand_points = np.vstack( (
                    b[:,0]*v1[0] + b[:,1]*v2[0] + b[:,2]*v3[0],
                    b[:,0]*v1[1] + b[:,1]*v2[1] + b[:,2]*v3[1]
                )).T
            for i in range(rand_points.shape[0]):
                b_ours = gpytoolbox.barycentric_coordinates(rand_points[i,:],v1,v2,v3)
                self.assertTrue(np.isclose(b[i,:]-b_ours,0).all())
                # All should be positive since we're in the triangle
                self.assertTrue((b_ours>=0).all())
            # Now test out of triangle: Generate random points in the triangle plane
            u = 5*np.random.rand(num_samples,2)-10
            v12 = v2 - v1
            v13 = v3 - v1
            rand_points = np.vstack( (
                    v1[0] + u[:,0]*v12[0] + u[:,1]*v13[0],
                    v1[1] + u[:,0]*v12[1] + u[:,1]*v13[1]
                )).T
            # Theses are not unique so require more care
            b_ours_big = np.zeros((num_samples,3))
            for i in range(rand_points.shape[0]):
                b_ours_big[i,:] = gpytoolbox.barycentric_coordinates(rand_points[i,:],v1,v2,v3)
                # print(b_ours_big[i,:])
            our_points = np.vstack( (
                    b_ours_big[:,0]*v1[0] + b_ours_big[:,1]*v2[0] + b_ours_big[:,2]*v3[0],
                    b_ours_big[:,0]*v1[1] + b_ours_big[:,1]*v2[1] + b_ours_big[:,2]*v3[1],
                )).T
            # print(our_points)
            # print(rand_points)
            self.assertTrue(np.isclose(our_points-rand_points,0.0).all())
    def test_out_of_plane(self):
        for j in range(100):
            # Generate random triangle
            num_samples = 100
            v1 = np.random.rand(3)
            v2 = np.random.rand(3)
            v3 = np.random.rand(3)
            s = np.random.rand(num_samples,1)
            t = np.random.rand(num_samples,1)
            out_of_plane_disp = np.random.rand(num_samples)
            # Normal direction to the triangle
            n = np.cross(v2-v1,v3-v1)
            #b = np.array([1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)])
            b = np.hstack( (1 - np.sqrt(t),(1-s)*np.sqrt(t),s*np.sqrt(t)) )
            rand_points = np.vstack( (
                    b[:,0]*v1[0] + b[:,1]*v2[0] + b[:,2]*v3[0],
                    b[:,0]*v1[1] + b[:,1]*v2[1] + b[:,2]*v3[1],
                    b[:,0]*v1[2] + b[:,1]*v2[2] + b[:,2]*v3[2]
                )).T
            for i in range(rand_points.shape[0]):
                b_ours = gpytoolbox.barycentric_coordinates(rand_points[i,:] + out_of_plane_disp[i]*n,v1,v2,v3)
                # Should be the same as if we had not displaced it
                self.assertTrue(np.isclose(b[i,:]-b_ours,0).all())


if __name__ == '__main__':
    unittest.main()