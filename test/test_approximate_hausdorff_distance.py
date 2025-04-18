import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestApproximateHausdorffDistance(unittest.TestCase):
    def test_identical_mesh(self):
        # base square mesh in 2D
        V2d, F = gpytoolbox.regular_square_mesh(20)
        # embed into 3D (z=0)
        V = np.hstack([V2d, np.zeros((V2d.shape[0], 1))])
        # identical triangular surface => zero distance in both backends
        for use_cpp in (False, True):
            d = gpytoolbox.approximate_hausdorff_distance(
                V, F, V, F, use_cpp=use_cpp
            )
            self.assertAlmostEqual(d, 0.0, places=7)

    def test_translated_mesh(self):
        V2d, F = gpytoolbox.regular_square_mesh(20)
        # embed into 3D (z=0)
        V = np.hstack([V2d, np.zeros((V2d.shape[0], 1))])
        # translate the surface by a 3D vector => distance = vector length
        shift = np.array([-0.7, 2.5, 1.1])
        U = V + shift
        expected = np.linalg.norm(shift)
        for use_cpp in (False, True):
            d = gpytoolbox.approximate_hausdorff_distance(
                V, F, U, F, use_cpp=use_cpp
            )
            self.assertAlmostEqual(d, expected, places=7)

    def test_bunny(self):
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        n = gpytoolbox.per_vertex_normals(V,F)
        np.random.seed(0)
        r = np.random.rand()
        u = V + r*n
        g = F.copy()
        dist1 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=False)
        dist2 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=True)
        # print(r)
        # print(dist1)
        # print(dist2)
        self.assertTrue(np.isclose(dist1,dist2))
        self.assertTrue(np.isclose(dist1,r))
    def test_bunny_smaller(self):
        # our bunny is too big for the python implementation, so we'll use a smaller one
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        V,F,_,_ = gpytoolbox.decimate(V,F,face_ratio=0.1)
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        n = gpytoolbox.per_vertex_normals(V,F)
        np.random.seed(0)
        for i in range(10):
            r = np.random.rand()
            u = V + r*n
            g = F.copy()
            dist1 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=False)
            dist2 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=True)
            # print(r)
            # print(dist1)
            # print(dist2)
            self.assertTrue(np.isclose(dist1,dist2))
            self.assertTrue(np.isclose(dist1,r))

    def test_identical_polylines(self):
        # identical polyline => distance = 0
        v = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [1.0, 1.0]])
        e = np.array([[0, 1],
                      [1, 2]])
        d = gpytoolbox.approximate_hausdorff_distance(v, e, v, e)
        self.assertAlmostEqual(d, 0.0, places=7)

    def test_reversed_polyline(self):
        # reversed ordering of vertices/edges still zero distance
        v1 = np.array([[0.0, 0.0],
                       [1.0, 0.0],
                       [2.0, 0.0]])
        e1 = np.array([[0, 1],
                       [1, 2]])
        v2 = v1[::-1]       # reverse list of points
        e2 = np.array([[2, 1],
                       [1, 0]])
        d = gpytoolbox.approximate_hausdorff_distance(v1, e1, v2, e2)
        self.assertAlmostEqual(d, 0.0, places=7)

    def test_line_segment_vs_longer_segment(self):
        # segment [0→1] vs [0→2], Hausdorff distance = 1
        v1 = np.array([[0.0, 0.0],
                       [1.0, 0.0]])
        e1 = np.array([[0, 1]])
        v2 = np.array([[0.0, 0.0],
                       [2.0, 0.0]])
        e2 = np.array([[0, 1]])
        d = gpytoolbox.approximate_hausdorff_distance(v1, e1, v2, e2)
        self.assertAlmostEqual(d, 1.0, places=7)

    def test_shifted_triangle(self):
        # triangle vs same triangle shifted by (1,1) => distance = sqrt(2)
        v1 = np.array([[0.0, 0.0],
                       [1.0, 0.0],
                       [0.0, 1.0]])
        e1 = np.array([[0, 1],
                       [1, 2],
                       [2, 0]])
        shift = np.array([1.0, 1.0])
        v2 = v1 + shift
        e2 = e1.copy()
        expected = np.linalg.norm(shift)
        d = gpytoolbox.approximate_hausdorff_distance(v1, e1, v2, e2)
        self.assertAlmostEqual(d, expected, places=7)

    def test_identical_circle(self):
        # identical circle => distance = 0
        V, E = gpytoolbox.regular_circle_polyline(50)
        d = gpytoolbox.approximate_hausdorff_distance(V, E, V, E)
        self.assertAlmostEqual(d, 0.0, places=7)

    def test_translated_circle(self):
        # translate circle by (dx,dy) => distance = sqrt(dx^2+dy^2)
        V, E = gpytoolbox.regular_circle_polyline(50)
        shift = np.array([0.3, -0.4])
        U = V + shift
        expected = np.linalg.norm(shift)
        d = gpytoolbox.approximate_hausdorff_distance(V, E, U, E)
        self.assertAlmostEqual(d, expected, places=7)

    

if __name__ == '__main__':
    unittest.main()
