import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestTriangleTriangleDistance(unittest.TestCase):
    # This isn't too complex, probably could use being expanded
    def test_synthetic(self):
        # Make two triangles whose distance we know
        P1 = np.array([0.0,0.0,0.0])
        Q1 = np.array([1.0,0.0,0.0])
        T1 = np.array([0.0,1.0,0.0])
        P2 = np.array([2.0,0.0,0.0])
        Q2 = np.array([3.0,0.0,0.0])
        T2 = np.array([2.0,1.0,0.0])
        dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
        # print(dist)
        self.assertTrue(np.isclose(dist,1.0))

        # Case 1: one of the closest points is a vertex and the other is interior to a face
        P1 = np.array([0.0,0.0,0.0])
        Q1 = np.array([1.0,0.0,0.0])
        T1 = np.array([0.0,1.0,0.0])

        P2 = np.array([0.25,0.25,1.0])
        Q2 = np.array([0.25,0.25,2.0])
        T2 = np.array([0.25,1.0,2.0])
        dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
        # print("R1 : ", R1)
        # print("R2 : ", R2)
        # print("dist : ", dist)
        self.assertTrue(np.isclose(dist,1.0))

        # Case 2: triangles are overlapping
        P1 = np.array([0.0,0.0,0.0])
        Q1 = np.array([1.0,0.0,0.0])
        T1 = np.array([0.0,1.0,0.0])
        P2 = np.array([0.25,0.25,-1.0])
        Q2 = np.array([0.25,0.25,2.0])
        T2 = np.array([0.25,1.0,2.0])
        dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
        self.assertTrue(np.isclose(dist,0.0))

        # Case 3: An edge of one triangle is parallel to the other triangle's face
        P1 = np.array([0.0,0.0,0.0])
        Q1 = np.array([1.0,0.0,0.0])
        T1 = np.array([0.0,1.0,0.0])

        P2 = np.array([0.25,-1,1.0])
        Q2 = np.array([0.25,1,1.0])
        T2 = np.array([0.25,0.0,2.0])
        dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
        # print("dist : ", dist)
        self.assertTrue(np.isclose(dist,1.0))

        # Case 4: Degenerate triangles (intersecting)
        P1 = np.array([0.0,0.0,0.0])
        Q1 = np.array([1.0,0.0,0.0])
        T1 = np.array([0.0,1.0,0.0])

        P2 = np.array([0.25,0.25,0.0])
        Q2 = np.array([0.25,2.0,0.0])
        T2 = np.array([0.25,0.0,0.0])
        dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
        # print("dist : ", dist)
        self.assertTrue(np.isclose(dist,0.0))

        # Case 5: Degenerate triangles (contained)
        P1 = np.array([0.0,0.0,0.0])
        Q1 = np.array([1.0,0.0,0.0])
        T1 = np.array([0.0,1.0,0.0])

        P2 = np.array([0.05,0.05,0.0])
        Q2 = np.array([0.05,0.05,0.0])
        T2 = np.array([0.05,0.1,0.0])
        dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
        # print("dist : ", dist)
        self.assertTrue(np.isclose(dist,0.0))




    def test_consistency_meshes(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
        np.random.seed(0)
        for mesh in meshes:
            v,f = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
            v = gpytoolbox.normalize_points(v)
            for i in range(100): 
                # Now pick a random triangle
                # Set random seed
                f1 = np.random.randint(f.shape[0])
                P1 = v[f[f1,0],:]
                Q1 = v[f[f1,1],:]
                T1 = v[f[f1,2],:]
                # Now pick a random edge
                f2 = np.random.randint(f.shape[0])
                P2 = v[f[f2,0],:]
                Q2 = v[f[f2,1],:]
                T2 = v[f[f2,2],:]
                dist = gpytoolbox.triangle_triangle_distance(P1,Q1,T1,P2,Q2,T2)
                self.is_correct_distance(P1,Q1,T1,P2,Q2,T2,dist)
    
    def is_correct_distance(self,P1,Q1,T1,P2,Q2,T2,dist):
        # Sample the first triangle densely
        V1 = np.zeros((3,3))
        V1[0,:] = P1
        V1[1,:] = Q1
        V1[2,:] = T1
        points_1 = gpytoolbox.random_points_on_mesh(V1,np.array([[0,1,2]]),1000)
        # Sample the second triangle densely
        V2 = np.zeros((3,3))
        V2[0,:] = P2
        V2[1,:] = Q2
        V2[2,:] = T2
        points_2 = gpytoolbox.random_points_on_mesh(V2,np.array([[0,1,2]]),1000)

        # Minimum distance between the rows of points_1 and points_2
        min_dist = np.min(np.linalg.norm(points_1[:,np.newaxis,:]-points_2[np.newaxis,:,:],axis=2))
        # print("min_dist :",min_dist)
        # print("dist :",dist)
        self.assertTrue(np.isclose(dist,min_dist,atol=1e-2))

        

if __name__ == '__main__':
    unittest.main()
