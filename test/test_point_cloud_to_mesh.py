from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
from scipy.stats import norm
import os

class TestPointCloudToMesh(unittest.TestCase):
    def test_circle(self):
        r = np.linspace(0, 2.*np.pi, num=100, endpoint=False)
        P = np.stack((np.cos(r), np.sin(r)), axis=-1)
        N = P.copy()

        V,F = gpy.point_cloud_to_mesh(P,N)

        self.assertTrue(np.isclose(np.linalg.norm(V, axis=-1), 1., rtol=1e-2, atol=1e-6).all())

    def test_sphere(self):
        P,_ = gpy.icosphere(3)
        N = P.copy()

        V,F = gpy.point_cloud_to_mesh(P,N)

        self.assertTrue(np.isclose(np.linalg.norm(V, axis=-1), 1., rtol=1e-2, atol=1e-6).all())

    def test_variety_of_meshes(self):
        rng = np.random.default_rng()
        meshes = ["spot.obj", "bunny_oded.obj", "armadillo.obj"]
        bdry_types = ["Neumann", "Dirichlet"]
        for bdry_type in bdry_types:
            for mesh in meshes:
                V0,F0 = gpy.read_mesh("test/unit_tests_data/" + mesh)
                V0 = gpy.normalize_points(V0)
                P,I,u = gpy.random_points_on_mesh(V0, F0, 20*V0.shape[0], rng=rng,
                    return_indices=True)
                N = gpy.per_face_normals(V0,F0)[I,:]

                V,F = gpy.point_cloud_to_mesh(P,N,
                    method='PSR',
                    psr_outer_boundary_type=bdry_type)

                h = gpy.approximate_hausdorff_distance(V0,F0,V,F)
                print(h)
                self.assertTrue(h<0.01)

if __name__ == '__main__':
    unittest.main()