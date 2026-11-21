from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


def _sphere(P, r=0.6):
    return np.linalg.norm(P, axis=1) - r


def _torus(P):
    q = np.linalg.norm(P[:, :2], axis=1) - 0.5
    return np.sqrt(q**2 + P[:, 2]**2) - 0.2


def _gyroid_ball(P):
    # Not an SDF: a gyroid clipped to a ball, full of ambiguous cells
    k = 9.0
    x, y, z = k * P[:, 0], k * P[:, 1], k * P[:, 2]
    g = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    return np.maximum(g, (np.linalg.norm(P, axis=1) - 0.8) * 10.0)


def _near_surface(f, L=1.0):
    # Split cells the level set of an L-Lipschitz f may cross
    def subdivide(lo, hi, depth):
        return np.abs(f(0.5 * (lo + hi))) <= L * 0.5 * np.linalg.norm(hi - lo, axis=1)
    return subdivide


def _directed_edges(F):
    return F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)


def _closed_oriented(F):
    # Every directed edge appears once, and so does its opposite
    E = _directed_edges(F)
    _, counts = np.unique(E, axis=0, return_counts=True)
    if np.any(counts != 1):
        return False
    a = {tuple(e) for e in E}
    return all((e[1], e[0]) in a for e in E)


def _open_edges(F):
    # Directed edges without an opposite
    a = {tuple(e) for e in _directed_edges(F)}
    return np.array([e for e in a if (e[1], e[0]) not in a]).reshape(-1, 2)


def _euler(F):
    E = np.unique(np.sort(_directed_edges(F), axis=1), axis=0)
    return len(np.unique(F)) - len(E) + len(F)


def _volume(V, F):
    return np.sum(np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]]))) / 6.0


def _normals(V, F):
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


class _OctreeMarchingCubesTests:
    # Run for each method by the TestCase subclasses at the bottom
    method = None

    def mc(self, *args, **kwargs):
        return gpy.octree_marching_cubes(*args, method=self.method, **kwargs)

    def test_uniform_sphere(self):
        V, F = self.mc(_sphere, lambda lo, hi, d: np.ones(len(d), dtype=bool),
                                         max_depth=4)
        self.assertTrue(F.shape[0] > 0)
        self.assertTrue(_closed_oriented(F))
        self.assertEqual(_euler(F), 2)
        exact = 4.0 / 3.0 * np.pi * 0.6**3
        self.assertTrue(abs(_volume(V, F) - exact) < 0.05 * exact)

    def test_adaptive_sphere_and_torus(self):
        for interpolation in ['linear', 'binary_search']:
            V, F = self.mc(_sphere, max_depth=7, interpolation=interpolation)
            self.assertTrue(_closed_oriented(F))
            self.assertEqual(_euler(F), 2)
            exact = 4.0 / 3.0 * np.pi * 0.6**3
            self.assertTrue(abs(_volume(V, F) - exact) < 0.005 * exact)

            V, F = self.mc(_torus, max_depth=7, interpolation=interpolation)
            self.assertTrue(_closed_oriented(F))
            self.assertEqual(_euler(F), 0)
            exact = 2 * np.pi**2 * 0.5 * 0.2**2
            self.assertTrue(abs(_volume(V, F) - exact) < 0.01 * exact)

    def test_level_jumps_all_orientations(self):
        # Leaves on one side of a plane through the sphere are 3 levels
        # smaller: closed, and every triangle faces the way the field grows
        coarse = self.mc(_sphere, lambda lo, hi, d: np.ones(len(d), dtype=bool),
                                           max_depth=3)
        coarse_error = abs(_volume(*coarse) - 4.0 / 3.0 * np.pi * 0.6**3)
        for face in range(6):
            a, sign = face // 2, (1.0 if face % 2 else -1.0)
            def subdivide(lo, hi, depth):
                mid = 0.5 * (lo[:, a] + hi[:, a])
                return depth < np.where(sign * mid > 0.1, 6, 3)
            for interpolation in ['linear', 'binary_search']:
                V, F = self.mc(_sphere, subdivide, max_depth=6,
                                                 interpolation=interpolation)
                self.assertTrue(_closed_oriented(F))
                self.assertEqual(_euler(F), 2)
                centroids = V[F].mean(axis=1)
                self.assertTrue(np.all(np.einsum('ij,ij->i', _normals(V, F), centroids) > 0))
                exact = 4.0 / 3.0 * np.pi * 0.6**3
                self.assertTrue(abs(_volume(V, F) - exact) < coarse_error)

    def test_mixed_depths(self):
        # Large depth jumps, which balancing must smooth out
        def subdivide(lo, hi, depth):
            target = np.where(lo[:, 0] >= 0, 6, np.where(lo[:, 1] >= 0, 2, 4))
            return depth < target
        for f in [_sphere, _torus, _gyroid_ball]:
            V, F = self.mc(f, subdivide, max_depth=6)
            self.assertTrue(F.shape[0] > 0)
            self.assertTrue(_closed_oriented(F))
            self.assertTrue(_volume(V, F) > 0)

    def test_gyroid(self):
        for depth in range(3, 7):
            V, F = self.mc(_gyroid_ball, _near_surface(_gyroid_ball, 30.0),
                                             max_depth=depth)
            self.assertTrue(F.shape[0] > 0)
            self.assertTrue(_closed_oriented(F))

    def test_checkerboard_around_edges(self):
        # Around the lines x = 0, z = 0, leaves in two diagonally opposite
        # quadrants are smaller than in the other two
        def subdivide(lo, hi, depth):
            fine = (lo[:, 0] >= 0) != (lo[:, 2] >= 0)
            return (depth < 2) | (depth < np.where(fine, 4, 3))
        plane = lambda P: P[:, 1] - 0.1
        tilted = lambda P: P[:, 1] - 0.2 * P[:, 0] + 0.15 * P[:, 2] - 0.05
        for interpolation in ['linear', 'binary_search']:
            for f in [plane, tilted]:
                V, F = self.mc(f, subdivide, max_depth=5,
                                                 interpolation=interpolation)
                self.assertEqual(gpy.non_manifold_edges(F).shape[0], 0)
                E = _open_edges(F)
                on_box = np.any(np.all(np.abs(np.abs(V[E]) - 1) < 1e-12, axis=1), axis=1)
                self.assertTrue(np.all(on_box))
            V, F = self.mc(_sphere, subdivide, max_depth=5,
                                             interpolation=interpolation)
            self.assertTrue(_closed_oriented(F))

    def test_plane_reaches_box(self):
        # A plane cut by the box, with level jumps: exact area, vertices on the
        # plane, normals towards +z
        plane = lambda P: P[:, 2] - 0.3 * P[:, 0] - 0.1
        def subdivide(lo, hi, depth):
            return depth < np.where(lo[:, 0] + lo[:, 1] < 0, 2, 6)
        for depth in [0, 1, 4, 6]:
            V, F = self.mc(plane, subdivide, max_depth=depth)
            self.assertTrue(np.isclose(np.sum(gpy.doublearea(V, F)) / 2, 4 * np.sqrt(1.09), atol=1e-9))
            self.assertTrue(np.all(np.abs(plane(V)) < 1e-12))
            N = _normals(V, F)
            nonflat = np.linalg.norm(N, axis=1) > 1e-12
            self.assertTrue(np.all(N[nonflat, 2] > 0))

    def test_binary_search_accuracy(self):
        # On uniform octrees (no vertex is moved by transition cells), a
        # vertex is within edge length / 2^(steps + 1) of the surface
        everywhere = lambda lo, hi, d: np.ones(len(d), dtype=bool)
        for steps in [5, 10, 20]:
            V, F = self.mc(_torus, everywhere, max_depth=5,
                                             interpolation='binary_search',
                                             binary_search_steps=steps)
            self.assertTrue(np.max(np.abs(_torus(V))) <= 2.0**(-4 - (steps + 1)))
        V_lin, _ = self.mc(_torus, everywhere, max_depth=3)
        V_bs, _ = self.mc(_torus, everywhere, max_depth=3,
                                            interpolation='binary_search', binary_search_steps=20)
        self.assertTrue(np.max(np.abs(_torus(V_bs))) < 1e-4 * np.max(np.abs(_torus(V_lin))))

    def test_isovalue_and_box(self):
        # The 0.1 level set of the sphere of radius 0.5 is the one of radius
        # 0.6; the root cell need not be a cube centered at the origin
        V, F = self.mc(lambda P: _sphere(P, 0.5), lo=(-0.7, -0.8, -0.9),
                                         hi=(0.9, 0.8, 0.7), max_depth=6, isovalue=0.1,
                                         interpolation='binary_search', binary_search_steps=30)
        self.assertTrue(_closed_oriented(F))
        self.assertTrue(np.max(np.abs(_sphere(V, 0.6))) < 1e-3)

    def test_mesh_sdf(self):
        # The signed distance to a closed mesh: watertight, and close to it
        V0, F0 = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        V0 = gpy.normalize_points(V0)
        sdf = lambda P: gpy.signed_distance(P, V0, F0)[0]
        V, F = self.mc(sdf, max_depth=7, interpolation='binary_search')
        self.assertTrue(_closed_oriented(F))
        dists = gpy.squared_distance(V, V0, F=F0, use_cpp=True)[0]
        self.assertTrue(np.all(np.sqrt(dists) < 2.0 / 2**7))

    def test_callbacks_are_batched(self):
        calls = {'sdf': 0, 'subdivide': 0}
        def sdf(P):
            calls['sdf'] += 1
            return _torus(P)
        near = _near_surface(_torus)
        def subdivide(lo, hi, depth):
            calls['subdivide'] += 1
            return near(lo, hi, depth) & (depth < np.where(lo[:, 0] < 0, 3, 6))
        V, F = self.mc(sdf, subdivide, max_depth=6,
                                         interpolation='binary_search', binary_search_steps=7)
        self.assertTrue(_closed_oriented(F))
        self.assertTrue(calls['subdivide'] <= 6)
        # Corners, 7 binary search steps, and (Transvoxel) the normals of the
        # vertices moved by transition cells
        self.assertEqual(calls['sdf'], 1 + 7 + (1 if self.method == 'transvoxel' else 0))

    def test_no_surface(self):
        V, F = self.mc(lambda P: np.ones(P.shape[0]), max_depth=3)
        self.assertEqual(V.shape, (0, 3))
        self.assertEqual(F.shape, (0, 3))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            self.mc(_sphere, max_depth=21)
        with self.assertRaises(ValueError):
            self.mc(_sphere, interpolation='cubic')
        with self.assertRaises(ValueError):
            gpy.octree_marching_cubes(_sphere, method='dual')
        with self.assertRaises(ValueError):
            self.mc(_sphere, lo=(1, -1, -1), hi=(-1, 1, 1))
        with self.assertRaises(ValueError):
            self.mc(lambda P: np.zeros(3), max_depth=2)
        with self.assertRaises(ValueError):
            self.mc(_sphere, lambda lo, hi, d: np.ones(len(d) + 1), max_depth=2)

    def test_moved_boundary_vertices_stay_on_plane(self):
        # A vertex on a root face, moved by a transition cell, stays on the
        # face and on the (planar) level set
        plane = lambda P: P[:, 2] - 0.3 * P[:, 0] - 0.2 * P[:, 1] - 0.1
        subdivide = lambda lo, hi, d: d < np.where(lo[:, 0] < 0, 2, 3)
        for interpolation in ['linear', 'binary_search']:
            V, F = self.mc(plane, subdivide, max_depth=3, interpolation=interpolation,
                           binary_search_steps=60)
            self.assertTrue(F.shape[0] > 0)
            self.assertTrue(np.max(np.abs(plane(V))) < 1e-12)
            E = _open_edges(F)
            on_box = np.any(np.all(np.abs(np.abs(V[E]) - 1) < 1e-12, axis=1), axis=1)
            self.assertTrue(np.all(on_box))

    def test_moved_vertices_stay_in_box(self):
        plane = lambda P: (P[:, 0] + P[:, 1] + P[:, 2] - 0.99) / np.sqrt(3)
        subdivide = lambda lo, hi, d: d < np.where(lo[:, 1] < 0, 1, 2)
        V, F = self.mc(plane, subdivide, max_depth=2)
        self.assertTrue(F.shape[0] > 0)
        self.assertTrue(np.all(np.abs(V) <= 1))
        self.assertTrue(np.max(np.abs(plane(V))) < 1e-12)

    def test_non_dyadic_box(self):
        # Root faces at non-dyadic coordinates: vertices on them stay there
        plane = lambda P: P[:, 2] - 0.3 * P[:, 0] - 0.2 * P[:, 1] - 0.1
        lo, hi = np.array([-0.7, -1.3, -0.9]), np.array([0.9, 0.3, 1.1])
        subdivide = lambda l, h, d: d < np.where(l[:, 0] < 0.1, 2, 4)
        V, F = self.mc(plane, subdivide, lo=lo, hi=hi, max_depth=4)
        self.assertTrue(np.max(np.abs(plane(V))) < 1e-12)
        self.assertTrue(np.all((V >= lo) & (V <= hi)))
        E = _open_edges(F)
        on_face = np.isclose(V[E][:, :, None, :], np.stack([lo, hi])[None, None], atol=1e-12, rtol=0)
        self.assertTrue(np.all(np.any(np.all(on_face, axis=1), axis=(1, 2))))

    def test_rotated_gyroids_are_manifold(self):
        # Rotated gyroids where the surface crosses a leaf face twice: two
        # leaves could triangulate their polygons with the same diagonal
        # (non-manifold fins with 'psr', before the fix)
        cases = [(7.660021, 2.057433, (-0.576190, 0.095881, -0.408982, -0.701103), 4),
                 (10.750334, 2.505081, (0.131600, -0.140939, 0.549842, -0.812706), 3),
                 (11.627310, 6.142166, (-0.527205, -0.043164, 0.521890, -0.669196), 3)]
        for k, phase, (w, x, y, z), depth in cases:
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            def f(P):
                Q = k * P @ R.T + phase
                g = (np.sin(Q[:, 0]) * np.cos(Q[:, 1]) + np.sin(Q[:, 1]) * np.cos(Q[:, 2]) +
                     np.sin(Q[:, 2]) * np.cos(Q[:, 0]))
                return np.maximum(g, (np.linalg.norm(P, axis=1) - 0.8) * 10)
            V, F = self.mc(f, lambda lo, hi, d: np.ones(len(d), dtype=bool), max_depth=depth)
            self.assertEqual(gpy.non_manifold_edges(F).shape[0], 0)
            self.assertTrue(_closed_oriented(F))


class TestOctreeMarchingCubesTransvoxel(_OctreeMarchingCubesTests, unittest.TestCase):
    method = 'transvoxel'


class TestOctreeMarchingCubesPSR(_OctreeMarchingCubesTests, unittest.TestCase):
    method = 'psr'


if __name__ == '__main__':
    unittest.main()
