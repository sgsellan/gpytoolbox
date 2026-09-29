from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


class TestDihedralAngles(unittest.TestCase):

    @staticmethod
    def _unit_cube():
        V = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float64)
        F = np.array([
            [0,2,1],[0,3,2],
            [4,5,6],[4,6,7],
            [0,1,5],[0,5,4],
            [1,2,6],[1,6,5],
            [2,3,7],[2,7,6],
            [3,0,4],[3,4,7]],dtype=np.int32)
        return V, F

    def test_single_triangle_all_boundary(self):
        # A single triangle has no interior edges, so every dihedral angle is
        # undefined (nan).
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
        f = np.array([[0,1,2]],dtype=np.int32)
        theta = gpy.dihedral_angles(v,f)
        self.assertEqual(theta.shape, (1,3))
        self.assertTrue(np.all(np.isnan(theta)))

    def test_flat_plane_is_zero(self):
        # A flat (planar) mesh has zero dihedral angle on all interior edges.
        v,f = gpy.regular_square_mesh(8)
        v3 = np.hstack([v, np.zeros((v.shape[0],1))])
        theta = gpy.dihedral_angles(v3, f.astype(np.int32))
        interior = ~np.isnan(theta)
        self.assertTrue(np.any(interior))
        self.assertTrue(np.allclose(theta[interior], 0.0, atol=1e-7))

    def test_cube_right_angles(self):
        # Adjacent faces of a cube meet either coplanar (the two triangles of a
        # face, angle 0) or at a right angle (across a cube edge, angle pi/2).
        V,F = self._unit_cube()
        theta = gpy.dihedral_angles(V,F)
        vals = theta[~np.isnan(theta)]
        # Every angle is either ~0 or ~pi/2.
        is_zero = np.isclose(vals, 0.0, atol=1e-7)
        is_right = np.isclose(vals, np.pi/2, atol=1e-7)
        self.assertTrue(np.all(is_zero | is_right))
        # There should be no boundary edges on a closed cube.
        self.assertFalse(np.any(np.isnan(theta)))

    def test_cube_thresholding_finds_12_edges(self):
        # Thresholding the dihedral angle at 45 degrees recovers exactly the 12
        # edges of the cube.
        V,F = self._unit_cube()
        theta = gpy.dihedral_angles(V,F)
        sharp = theta > np.deg2rad(45.0)
        det = []
        for c in range(3):
            mask = sharp[:,c]
            if np.any(mask):
                det.append(np.stack([F[mask,(c+1)%3], F[mask,(c+2)%3]], axis=1))
        det = np.unique(np.sort(np.concatenate(det,axis=0),axis=1),axis=0)
        expected = np.unique(np.sort(np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]]),axis=1),axis=0)
        self.assertEqual(det.shape[0], 12)
        self.assertTrue(np.array_equal(det, expected))

    def test_interior_edge_symmetric(self):
        # An interior edge is shared by two faces; both halfedge entries should
        # carry the same angle.
        np.random.seed(0)
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        f = f.astype(np.int32)
        theta = gpy.dihedral_angles(v,f)
        TT,TTi = gpy.triangle_triangle_adjacency(f)
        m = f.shape[0]
        for c in range(3):
            nb = TT[:,c]
            valid = nb >= 0
            here = theta[valid, c]
            there = theta[nb[valid], TTi[valid, c]]
            self.assertTrue(np.allclose(here, there, atol=1e-9, equal_nan=True))
        # All interior angles are within [0, pi].
        interior = theta[~np.isnan(theta)]
        self.assertTrue(np.all(interior >= -1e-9))
        self.assertTrue(np.all(interior <= np.pi + 1e-9))


if __name__ == '__main__':
    unittest.main()
