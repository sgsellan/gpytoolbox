from .context import numpy as np
from .context import unittest
from .context import gpytoolbox as gpy
from gpytoolbox.matryoshka import _feasible, _transform_points, _sample_B_surface


def _unit_cube():
    V = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=np.float64)
    F = np.array([
        [0,2,1],[0,3,2],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [2,3,7],[2,7,6],
        [1,2,6],[1,6,5],
        [0,4,7],[0,7,3],
    ], dtype=np.int32)
    return V, F


def _verify_feasible(V, F, res, n_samples=300, seed=1):
    """Re-check the returned configuration with the feasibility test."""
    rng = np.random.default_rng(seed)
    samples_B = _sample_B_surface(V, F.astype(np.int32), n_samples, rng)
    samples_TB = _transform_points(samples_B, res['s'], res['R'],
                                   res['c'], res['B_center'])
    return _feasible(samples_TB, V, F.astype(np.int32),
                     res['cut_point'], res['cut_normal'],
                     res['a_plus'], res['a_minus'])


class TestMatryoshka(unittest.TestCase):

    def test_sphere_scale_only(self):
        # A self-nesting sphere with identity rotation centered at the origin
        # should accept very nearly the full unit scale.
        V, F = gpy.icosphere(2)
        res = gpy.matryoshka(V, F, optimize='scale_only',
                             n_samples=300, seed=0)
        self.assertGreater(res['s'], 0.95)
        self.assertTrue(_verify_feasible(V, F, res))

    def test_cube_scale_only(self):
        # Axis-aligned cube self-nesting with horizontal cut should also
        # accept ~full scale (the two halves slide cleanly off along ±z).
        V, F = _unit_cube()
        res = gpy.matryoshka(V, F, optimize='scale_only',
                             n_samples=300, seed=0)
        self.assertGreater(res['s'], 0.95)
        self.assertTrue(_verify_feasible(V, F, res))

    def test_sphere_rigid(self):
        # With rigid (scale + rotation + centroid) optimization, the sphere
        # should still find a near-full nesting given a modest budget.
        V, F = gpy.icosphere(2)
        np.random.seed(0)
        res = gpy.matryoshka(V, F, optimize='rigid',
                             n_samples=80, n_particles=30, max_iter=40,
                             seed=0)
        self.assertGreater(res['s'], 0.7)
        self.assertTrue(_verify_feasible(V, F, res))

    def test_sphere_all(self):
        # Full optimization (16 free params) is the hardest case. We only
        # require a non-trivial feasible nesting on the sanity-check budget.
        V, F = gpy.icosphere(2)
        np.random.seed(0)
        res = gpy.matryoshka(V, F, optimize='all',
                             n_samples=80, n_particles=25, max_iter=25,
                             seed=0)
        self.assertGreater(res['s'], 0.4)
        self.assertTrue(_verify_feasible(V, F, res))

    def test_warm_started_all_beats_rigid(self):
        # Warm-starting 'all' from a precomputed rigid result must not lose
        # ground vs that rigid baseline (modulo the binary-search tolerance).
        # We use a single rigid run as the baseline and pass it in so the
        # comparison is against a *fixed* number, not against another
        # non-deterministic rigid call.
        V, F = gpy.icosphere(2)
        np.random.seed(0)
        r_rigid = gpy.matryoshka(V, F, optimize='rigid',
                                 n_samples=100, n_particles=50, max_iter=50,
                                 seed=0)
        # Re-evaluate the rigid config with the same outer-call seed, so the
        # baseline number is reproduced deterministically.
        r_baseline = gpy.matryoshka(
            V, F, optimize='scale_only', n_samples=100, seed=0,
            R=r_rigid['R'], c=r_rigid['c'],
            cut_point=r_rigid['cut_point'], cut_normal=r_rigid['cut_normal'],
            a_plus=r_rigid['a_plus'], a_minus=r_rigid['a_minus'])

        for trial in range(3):
            np.random.seed(trial)
            r_all = gpy.matryoshka(V, F, optimize='all',
                                   n_samples=100, n_particles=50, max_iter=50,
                                   seed=0, warm_start=r_rigid)
            # Allow a small slack for the inner binary-search tolerance.
            self.assertGreaterEqual(r_all['s'] + 1e-3, r_baseline['s'])
            self.assertTrue(_verify_feasible(V, F, r_all))

    def test_returned_keys(self):
        V, F = gpy.icosphere(1)
        res = gpy.matryoshka(V, F, optimize='scale_only',
                             n_samples=50, seed=0)
        for key in ('s', 'R', 'c', 'B_center',
                    'cut_point', 'cut_normal', 'a_plus', 'a_minus'):
            self.assertIn(key, res)
        self.assertEqual(res['R'].shape, (3, 3))
        self.assertEqual(res['c'].shape, (3,))
        self.assertEqual(res['cut_normal'].shape, (3,))

    def test_disparate_nesting(self):
        # Nest a smaller sphere (B) inside a larger sphere (A). The optimal
        # scale should obviously be very close to 1 because the smaller B
        # fits trivially inside A.
        VA, FA = gpy.icosphere(2)
        VB, FB = gpy.icosphere(1)
        VB = 0.3 * VB
        res = gpy.matryoshka(VA, FA, VB=VB, FB=FB, optimize='scale_only',
                             n_samples=200, seed=0)
        self.assertGreater(res['s'], 0.95)


    def test_bunny_warm_started_beats_rigid(self):
        # Real shape: bunny self-nesting. Warm-started 'all' should not lose
        # ground vs the rigid baseline.
        V, F = gpy.read_mesh('test/unit_tests_data/bunny_oded.obj')
        V = V - V.mean(0)
        V = V / np.max(np.abs(V))   # normalize to unit-ish

        np.random.seed(0)
        r_rigid = gpy.matryoshka(V, F, optimize='rigid',
                                 n_samples=80, n_particles=20, max_iter=20,
                                 seed=0)
        # Re-evaluate to get a deterministic baseline number.
        r_baseline = gpy.matryoshka(
            V, F, optimize='scale_only', n_samples=80, seed=0,
            R=r_rigid['R'], c=r_rigid['c'],
            cut_point=r_rigid['cut_point'], cut_normal=r_rigid['cut_normal'],
            a_plus=r_rigid['a_plus'], a_minus=r_rigid['a_minus'])

        np.random.seed(0)
        r_all = gpy.matryoshka(V, F, optimize='all',
                               n_samples=80, n_particles=20, max_iter=20,
                               seed=0, warm_start=r_rigid)
        self.assertGreater(r_baseline['s'], 0.1,
                           msg="rigid baseline collapsed — algorithm broken?")
        self.assertGreaterEqual(r_all['s'] + 1e-3, r_baseline['s'])
        self.assertTrue(_verify_feasible(V, F, r_all))

    def test_teddy_warm_started_beats_rigid(self):
        # Same invariant on the teddy mesh.
        V, F = gpy.read_mesh('test/unit_tests_data/teddy.obj')
        V = V - V.mean(0)
        V = V / np.max(np.abs(V))

        np.random.seed(0)
        r_rigid = gpy.matryoshka(V, F, optimize='rigid',
                                 n_samples=80, n_particles=20, max_iter=20,
                                 seed=0)
        r_baseline = gpy.matryoshka(
            V, F, optimize='scale_only', n_samples=80, seed=0,
            R=r_rigid['R'], c=r_rigid['c'],
            cut_point=r_rigid['cut_point'], cut_normal=r_rigid['cut_normal'],
            a_plus=r_rigid['a_plus'], a_minus=r_rigid['a_minus'])

        np.random.seed(0)
        r_all = gpy.matryoshka(V, F, optimize='all',
                               n_samples=80, n_particles=20, max_iter=20,
                               seed=0, warm_start=r_rigid)
        self.assertGreater(r_baseline['s'], 0.1,
                           msg="rigid baseline collapsed — algorithm broken?")
        self.assertGreaterEqual(r_all['s'] + 1e-3, r_baseline['s'])
        self.assertTrue(_verify_feasible(V, F, r_all))


if __name__ == '__main__':
    unittest.main()
