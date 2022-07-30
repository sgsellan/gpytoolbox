from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestPerVertexNormals(unittest.TestCase):
    # TODO WRITE THIS WITHOUT IGL
    def test_against_igl(self):
        self.assertTrue(True)
    #     # Trying several meshes against the libigl per_vertex_normals
    #     v,f = igl.read_triangle_mesh("test/unit_tests_data/bunny_oded.obj")
    #     n_gt = igl.per_vertex_normals(v,f,weighting=1)
    #     n_gt = n_gt/np.tile(np.linalg.norm(n_gt,axis=1)[:,None],(1,3))
    #     n = gpytoolbox.per_vertex_normals(v,f)
    #     self.assertTrue(np.isclose(n-n_gt,0.0).all())
    #     v,f = igl.read_triangle_mesh("test/unit_tests_data/armadillo.obj")
    #     n_gt = igl.per_vertex_normals(v,f,weighting=1)
    #     n_gt = n_gt/np.tile(np.linalg.norm(n_gt,axis=1)[:,None],(1,3))
    #     n = gpytoolbox.per_vertex_normals(v,f)
    #     self.assertTrue(np.isclose(n-n_gt,0.0).all())
    #     v,f = igl.read_triangle_mesh("test/unit_tests_data/cube.obj")
    #     n_gt = igl.per_vertex_normals(v,f,weighting=1)
    #     n_gt = n_gt/np.tile(np.linalg.norm(n_gt,axis=1)[:,None],(1,3))
    #     n = gpytoolbox.per_vertex_normals(v,f)
    #     self.assertTrue(np.isclose(n-n_gt,0.0).all())
    #     v,f = igl.read_triangle_mesh("test/unit_tests_data/mountain.obj")
    #     n_gt = igl.per_vertex_normals(v,f,weighting=1)
    #     n_gt = n_gt/np.tile(np.linalg.norm(n_gt,axis=1)[:,None],(1,3))
    #     n = gpytoolbox.per_vertex_normals(v,f)
    #     self.assertTrue(np.isclose(n-n_gt,0.0).all())


if __name__ == '__main__':
    unittest.main()