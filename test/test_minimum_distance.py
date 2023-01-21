import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestMinimumDistance(unittest.TestCase):
    def test_cube(self):
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        U,G = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        # Normalize mesh
        U = gpytoolbox.normalize_points(U,center=np.array([0.5,0.5,0.5]))
        random_displacements = 6*np.random.rand(20)
        for i in range(20):
            for j in range(3):
                U2 = U.copy()
                U2[:,j] += random_displacements[i]
                dist = gpytoolbox.minimum_distance(V,F,U2,G)
                # self.assertTrue(np.isclose(dist,0.0,atol=1e-2))
                dist_gt = np.clip(random_displacements[i]-1,0,np.Inf)
                # print(dist_gt,dist)
                self.assertTrue(np.isclose(dist,dist_gt,atol=1e-4))
    def test_bunny_faces(self):
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        n = gpytoolbox.per_face_normals(V,F,unit_norm=True)
        np.random.seed(0)
        for i in range(20):
            tiny_mesh_v = np.zeros((3,3))
            tiny_mesh_f = np.array([[0,1,2]])
            # random face
            f = np.random.randint(F.shape[0])
            tiny_mesh_v[0,:] = V[F[f,0],:] + 0.001*n[f,:]
            tiny_mesh_v[1,:] = V[F[f,1],:] + 0.001*n[f,:]
            tiny_mesh_v[2,:] = V[F[f,2],:] + 0.001*n[f,:]
            dist = gpytoolbox.minimum_distance(V,F,tiny_mesh_v,tiny_mesh_f)
            # print(dist)
            self.assertTrue(np.isclose(dist,0.001,atol=1e-2))
            
    # It would be nice to have more principled tests here...
    

if __name__ == '__main__':
    unittest.main()
