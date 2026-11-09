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


class TestOctreeMarchingCubes(unittest.TestCase):
    def test_uniform_sphere(self):
        V, F = gpy.octree_marching_cubes(_sphere, lambda lo, hi, d: np.ones(len(d), dtype=bool),
                                         max_depth=4)
        self.assertTrue(F.shape[0] > 0)
        self.assertTrue(_closed_oriented(F))
        self.assertEqual(_euler(F), 2)
        exact = 4.0 / 3.0 * np.pi * 0.6**3
        self.assertTrue(abs(_volume(V, F) - exact) < 0.05 * exact)

    def test_adaptive_sphere_and_torus(self):
        for interpolation in ['linear', 'binary_search']:
            V, F = gpy.octree_marching_cubes(_sphere, max_depth=7, interpolation=interpolation)
            self.assertTrue(_closed_oriented(F))
            self.assertEqual(_euler(F), 2)
            exact = 4.0 / 3.0 * np.pi * 0.6**3
            self.assertTrue(abs(_volume(V, F) - exact) < 0.005 * exact)

            V, F = gpy.octree_marching_cubes(_torus, max_depth=7, interpolation=interpolation)
            self.assertTrue(_closed_oriented(F))
            self.assertEqual(_euler(F), 0)
            exact = 2 * np.pi**2 * 0.5 * 0.2**2
            self.assertTrue(abs(_volume(V, F) - exact) < 0.01 * exact)

    def test_level_jumps_all_orientations(self):
        # Leaves on one side of a plane through the sphere are 3 levels
        # smaller: closed, and every triangle faces the way the field grows
        coarse = gpy.octree_marching_cubes(_sphere, lambda lo, hi, d: np.ones(len(d), dtype=bool),
                                           max_depth=3)
        coarse_error = abs(_volume(*coarse) - 4.0 / 3.0 * np.pi * 0.6**3)
        for face in range(6):
            a, sign = face // 2, (1.0 if face % 2 else -1.0)
            def subdivide(lo, hi, depth):
                mid = 0.5 * (lo[:, a] + hi[:, a])
                return depth < np.where(sign * mid > 0.1, 6, 3)
            for interpolation in ['linear', 'binary_search']:
                V, F = gpy.octree_marching_cubes(_sphere, subdivide, max_depth=6,
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
            V, F = gpy.octree_marching_cubes(f, subdivide, max_depth=6)
            self.assertTrue(F.shape[0] > 0)
            self.assertTrue(_closed_oriented(F))
            self.assertTrue(_volume(V, F) > 0)

    def test_gyroid(self):
        for depth in range(3, 7):
            V, F = gpy.octree_marching_cubes(_gyroid_ball, _near_surface(_gyroid_ball, 30.0),
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
                V, F = gpy.octree_marching_cubes(f, subdivide, max_depth=5,
                                                 interpolation=interpolation)
                self.assertEqual(gpy.non_manifold_edges(F).shape[0], 0)
                E = _open_edges(F)
                on_box = np.any(np.all(np.abs(np.abs(V[E]) - 1) < 1e-12, axis=1), axis=1)
                self.assertTrue(np.all(on_box))
            V, F = gpy.octree_marching_cubes(_sphere, subdivide, max_depth=5,
                                             interpolation=interpolation)
            self.assertTrue(_closed_oriented(F))

    def test_plane_reaches_box(self):
        # A plane cut by the box, with level jumps: exact area, vertices on the
        # plane, normals towards +z
        plane = lambda P: P[:, 2] - 0.3 * P[:, 0] - 0.1
        def subdivide(lo, hi, depth):
            return depth < np.where(lo[:, 0] + lo[:, 1] < 0, 2, 6)
        for depth in [0, 1, 4, 6]:
            V, F = gpy.octree_marching_cubes(plane, subdivide, max_depth=depth)
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
            V, F = gpy.octree_marching_cubes(_torus, everywhere, max_depth=5,
                                             interpolation='binary_search',
                                             binary_search_steps=steps)
            self.assertTrue(np.max(np.abs(_torus(V))) <= 2.0**(-4 - (steps + 1)))
        V_lin, _ = gpy.octree_marching_cubes(_torus, everywhere, max_depth=3)
        V_bs, _ = gpy.octree_marching_cubes(_torus, everywhere, max_depth=3,
                                            interpolation='binary_search', binary_search_steps=20)
        self.assertTrue(np.max(np.abs(_torus(V_bs))) < 1e-4 * np.max(np.abs(_torus(V_lin))))

    def test_isovalue_and_box(self):
        # The 0.1 level set of the sphere of radius 0.5 is the one of radius
        # 0.6; the root cell need not be a cube centered at the origin
        V, F = gpy.octree_marching_cubes(lambda P: _sphere(P, 0.5), lo=(-0.7, -0.8, -0.9),
                                         hi=(0.9, 0.8, 0.7), max_depth=6, isovalue=0.1,
                                         interpolation='binary_search', binary_search_steps=30)
        self.assertTrue(_closed_oriented(F))
        self.assertTrue(np.max(np.abs(_sphere(V, 0.6))) < 1e-3)

    def test_mesh_sdf(self):
        # The signed distance to a closed mesh: watertight, and close to it
        V0, F0 = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        V0 = gpy.normalize_points(V0)
        sdf = lambda P: gpy.signed_distance(P, V0, F0)[0]
        V, F = gpy.octree_marching_cubes(sdf, max_depth=7, interpolation='binary_search')
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
        V, F = gpy.octree_marching_cubes(sdf, subdivide, max_depth=6,
                                         interpolation='binary_search', binary_search_steps=7)
        self.assertTrue(_closed_oriented(F))
        self.assertTrue(calls['subdivide'] <= 6)
        # Corners, 7 binary search steps, normals of the moved vertices
        self.assertEqual(calls['sdf'], 1 + 7 + 1)

    def test_no_surface(self):
        V, F = gpy.octree_marching_cubes(lambda P: np.ones(P.shape[0]), max_depth=3)
        self.assertEqual(V.shape, (0, 3))
        self.assertEqual(F.shape, (0, 3))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            gpy.octree_marching_cubes(_sphere, max_depth=21)
        with self.assertRaises(ValueError):
            gpy.octree_marching_cubes(_sphere, interpolation='cubic')
        with self.assertRaises(ValueError):
            gpy.octree_marching_cubes(_sphere, lo=(1, -1, -1), hi=(-1, 1, 1))
        with self.assertRaises(ValueError):
            gpy.octree_marching_cubes(lambda P: np.zeros(3), max_depth=2)
        with self.assertRaises(ValueError):
            gpy.octree_marching_cubes(_sphere, lambda lo, hi, d: np.ones(len(d) + 1), max_depth=2)


if __name__ == '__main__':
    unittest.main()
