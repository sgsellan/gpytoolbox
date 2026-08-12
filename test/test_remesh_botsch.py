from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

# TODO: Check that projection is happening (with distances to surface, maybe?)

class TestRemeshBotsch(unittest.TestCase):
    def test_bunny(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True)
        # igl.write_obj("output.obj",u,g)
        E,bd = gpytoolbox.edges(g,return_boundary_indices=True)
        # Boundary should be empty
        self.assertTrue(len(bd)==0)
        # Edge lengths should be "near" 0.01
        edge_lengths = np.linalg.norm(u[E[0,:],:] - u[E[1,:],:],axis=1)
        self.assertTrue(np.isclose(np.mean(edge_lengths)-0.01,0.0,atol=1e-3))

    # This example used to break the remesher     
    def test_chair_example(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/wooden-chair-remesher-bug.obj")
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),5,0.1,True)
        # There used to be a bunch of duplicate vertices
        sv,_,_ = gpytoolbox.remove_duplicate_vertices(u)
        # There shouldn't be now
        self.assertTrue(u.shape[0]==sv.shape[0])
        # Same thing without projection
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),5,0.1,False)
        # There used to be a bunch of duplicate vertices
        sv,_,_ = gpytoolbox.remove_duplicate_vertices(u)
        # There shouldn't be now
        self.assertTrue(u.shape[0]==sv.shape[0])

    def test_with_boundary(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/airplane.obj")
        ind = gpytoolbox.boundary_vertices(f)
        boundary_verts = v[ind,:]
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),1,0.1,True)
        # gpytoolbox.write_mesh("test/unit_tests_data/airplane_output.obj",u,g)
        ind_output = gpytoolbox.boundary_vertices(g)
        boundary_verts_output = u[ind_output,:]
        # Boundary vertices should not move
        for i in range(len(ind)):
            dist = np.min(np.linalg.norm(np.tile(boundary_verts[i,:][None,:],(boundary_verts_output.shape[0],1)) - boundary_verts_output,axis=1))
            self.assertTrue(dist==0.0)

    def test_with_unique_features_closed(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # pick random faces of the model that are fixed
        feature = f[np.random.choice(range(f.shape[0]), v.shape[0]//1000, replace=False)].flatten()
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True,feature=feature)
        self.assertTrue(np.allclose(v[feature], u[:feature.shape[0]]))

    def test_with_unique_features(self):
        np.random.seed(0)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny.obj")
        # pick random faces of the model that are fixed
        feature = f[np.random.choice(range(f.shape[0]), v.shape[0]//1000, replace=False)].flatten()
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True,feature=feature)
        self.assertTrue(np.allclose(v[feature], u[:feature.shape[0]]))

    def test_with_not_unique_features(self):
        np.random.seed(8)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny.obj")
        # pick random faces of the model that are fixed
        feature = f[np.random.choice(range(f.shape[0]), v.shape[0]//1000, replace=False)].flatten()
        # check that they are not unique
        self.assertFalse(feature.shape[0] == np.unique(feature).shape[0])
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True,feature=feature)
        # unique feature nodes
        tmp, ind = np.unique(feature, return_index=True)
        feature_unique = tmp[np.argsort(ind)]
        self.assertTrue(np.allclose(v[feature_unique], u[:feature_unique.shape[0]]))

    def test_with_not_unique_features_and_boundary(self):
        np.random.seed(8)
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny.obj")
        # pick random faces of the model that are fixed and add some boundary nodes
        feature = f[np.random.choice(range(f.shape[0]), v.shape[0]//1000, replace=False)].flatten()
        feature = np.concatenate((feature, np.random.choice(gpytoolbox.boundary_vertices(f), 20, replace=False)))
        # check that they are not unique
        self.assertFalse(feature.shape[0] == np.unique(feature).shape[0])
        u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True,feature=feature)
        # unique feature nodes
        tmp, ind = np.unique(feature, return_index=True)
        feature_unique = tmp[np.argsort(ind)]
        self.assertTrue(np.allclose(v[feature_unique], u[:feature_unique.shape[0]]))

    @staticmethod
    def _unit_cube():
        # Surface mesh of the unit cube: 8 corners, 12 triangles, and the 12
        # cube edges (as vertex-index pairs) marking the sharp creases.
        V = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float64)
        F = np.array([
            [0,2,1],[0,3,2],   # z=0
            [4,5,6],[4,6,7],   # z=1
            [0,1,5],[0,5,4],   # y=0
            [1,2,6],[1,6,5],   # x=1
            [2,3,7],[2,7,6],   # y=1
            [3,0,4],[3,4,7]],dtype=np.int32) # x=0
        feature_edges = np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]],dtype=np.int32)
        return V, F, feature_edges

    def test_cube_feature_edges_isotropic(self):
        # Remeshing a cube with its 12 edges marked as feature edges should
        # produce an isotropic mesh whose vertices all lie on the cube surface,
        # with the 8 corners preserved and the creases kept sharp.
        np.random.seed(0)
        V, F, feature_edges = self._unit_cube()
        corners = V.copy()
        for h in [0.2, 0.1]:
            u, g = gpytoolbox.remesh_botsch(
                V.copy(), F.copy(), 20, h, True, feature_edges=feature_edges)

            # Closed manifold (no boundary).
            E, bd = gpytoolbox.edges(g, return_boundary_indices=True)
            self.assertEqual(len(bd), 0)

            # Every output vertex lies on the cube surface (some coordinate is
            # exactly 0 or 1).
            dist_to_surface = np.minimum.reduce([
                np.abs(u[:,0]-0), np.abs(u[:,0]-1),
                np.abs(u[:,1]-0), np.abs(u[:,1]-1),
                np.abs(u[:,2]-0), np.abs(u[:,2]-1)])
            self.assertTrue(np.max(dist_to_surface) < 1e-9)

            # Output stays inside the unit cube.
            self.assertTrue(np.all(u > -1e-9) and np.all(u < 1+1e-9))

            # The 8 corners survive exactly.
            for c in corners:
                self.assertTrue(np.min(np.linalg.norm(u-c, axis=1)) < 1e-9)

            # Edge lengths are isotropic and close to the target.
            edge_lengths = np.linalg.norm(u[E[:,0],:] - u[E[:,1],:], axis=1)
            self.assertTrue(np.isclose(np.mean(edge_lengths), h, atol=0.25*h))
            # The vast majority of edges fall within [0.5h, 1.5h].
            in_band = np.mean((edge_lengths > 0.5*h) & (edge_lengths < 1.5*h))
            self.assertTrue(in_band > 0.9)

    def test_cube_feature_edges_stay_sharp(self):
        # Each of the 12 cube edges should remain a sharp, straight crease: all
        # the vertices that subdivide it must lie exactly on the original edge.
        np.random.seed(0)
        V, F, feature_edges = self._unit_cube()
        u, g = gpytoolbox.remesh_botsch(
            V.copy(), F.copy(), 20, 0.1, True, feature_edges=feature_edges)
        for e in feature_edges:
            a, b = V[e[0]], V[e[1]]
            d = b - a
            count = 0
            for p in u:
                t = np.dot(p-a, d)/np.dot(d, d)
                if -1e-9 <= t <= 1+1e-9 and np.linalg.norm(a + t*d - p) < 1e-9:
                    count += 1
            # The two endpoints plus interior subdivision vertices: at least the
            # endpoints must be present, and a refined edge has several.
            self.assertTrue(count >= 2)

    def test_cube_detect_feature_edges(self):
        # Auto-detecting feature edges from the dihedral angle should recover
        # the cube's creases and give the same result as passing them manually.
        np.random.seed(0)
        V, F, feature_edges = self._unit_cube()
        corners = V.copy()
        u_manual, g_manual = gpytoolbox.remesh_botsch(
            V.copy(), F.copy(), 20, 0.1, True, feature_edges=feature_edges)
        u_auto, g_auto = gpytoolbox.remesh_botsch(
            V.copy(), F.copy(), 20, 0.1, True, detect_feature_edges=True)
        # Same topology and geometry as the manual feature-edge run.
        self.assertEqual(u_auto.shape, u_manual.shape)
        self.assertEqual(g_auto.shape, g_manual.shape)
        self.assertTrue(np.allclose(u_auto, u_manual))
        # Vertices stay on the cube surface and corners are preserved.
        dist_to_surface = np.minimum.reduce([
            np.abs(u_auto[:,0]-0), np.abs(u_auto[:,0]-1),
            np.abs(u_auto[:,1]-0), np.abs(u_auto[:,1]-1),
            np.abs(u_auto[:,2]-0), np.abs(u_auto[:,2]-1)])
        self.assertTrue(np.max(dist_to_surface) < 1e-9)
        for c in corners:
            self.assertTrue(np.min(np.linalg.norm(u_auto-c, axis=1)) < 1e-9)

    def test_cube_detect_feature_edges_high_threshold(self):
        # With a threshold above the cube's 90-degree creases, nothing is
        # detected, so the cube is free to lose its sharp edges (the result
        # differs from the crease-preserving run).
        np.random.seed(0)
        V, F, feature_edges = self._unit_cube()
        u_sharp, _ = gpytoolbox.remesh_botsch(
            V.copy(), F.copy(), 20, 0.2, True, detect_feature_edges=True,
            feature_dihedral_threshold=45.0)
        u_none, _ = gpytoolbox.remesh_botsch(
            V.copy(), F.copy(), 20, 0.2, True, detect_feature_edges=True,
            feature_dihedral_threshold=120.0)
        # The high-threshold run detects no features, so the two outputs should
        # not be identical in size/shape.
        self.assertFalse(u_sharp.shape == u_none.shape and np.allclose(u_sharp, u_none))

    def test_cube_feature_edges_arbitrary_refinement(self):
        # Finer target edge lengths yield monotonically more vertices.
        np.random.seed(0)
        V, F, feature_edges = self._unit_cube()
        counts = []
        for h in [0.4, 0.2, 0.1, 0.05]:
            u, g = gpytoolbox.remesh_botsch(
                V.copy(), F.copy(), 20, h, True, feature_edges=feature_edges)
            counts.append(u.shape[0])
        for i in range(1, len(counts)):
            self.assertTrue(counts[i] > counts[i-1])

    def test_nonmanifold_segfault(self):
        f = np.array([[0,1,2],[0,2,3],[2,0,4]],dtype=int)
        # choose a random v
        v = np.random.rand(5,3)
        # call remesh_botsch, this used to segfault
        # assert that it raises a ValueError
        with self.assertRaises(ValueError):
            u,g = gpytoolbox.remesh_botsch(v,f.astype(np.int32),20,0.01,True)
        # should not segfault

    # ------------------------------------------------------------------
    # Regression tests for feature-edge junction ("corner") preservation.
    #
    # These guard a bug where collapsing a feature edge silently lowered the
    # feature-edge degree of a crease junction, so the junction was mistaken
    # for a degree-2 "line" vertex and slid along the crease, cutting
    # ("chamfering") the corner off. Output vertices all stayed *on* the input
    # surface, so the only way to catch it is to check that (a) the original
    # sharp corners survive and (b) face interiors do not poke across the solid
    # (which also shows up as a change in enclosed volume).
    # ------------------------------------------------------------------
    @staticmethod
    def _extruded_prism(poly_xy, zlo=-0.5, zhi=0.5):
        # Prism over a 2D convex (CCW) polygon: side walls plus a top and a
        # bottom cap. Every original vertex is a sharp degree-3 corner.
        poly = np.asarray(poly_xy, dtype=np.float64)
        n = poly.shape[0]
        bot = np.column_stack([poly, np.full(n, zlo)])
        top = np.column_stack([poly, np.full(n, zhi)])
        V = np.vstack([bot, top])
        F = []
        for i in range(n):
            j = (i + 1) % n
            F.append([i, j, n + j])
            F.append([i, n + j, n + i])
        for i in range(1, n - 1):           # fan triangulation of the caps
            F.append([0, i + 1, i])         # bottom cap (faces -z)
            F.append([n, n + i, n + i + 1]) # top cap (faces +z)
        return V, np.array(F, dtype=np.int32)

    @staticmethod
    def _orient_outward(V, F):
        # Flip any face whose normal points toward the centroid (valid for the
        # convex polyhedra below), giving a consistently outward orientation.
        c = V.mean(0)
        F = F.copy()
        for i in range(F.shape[0]):
            a, b, cc = V[F[i, 0]], V[F[i, 1]], V[F[i, 2]]
            if np.dot(np.cross(b - a, cc - a), (a + b + cc) / 3.0 - c) < 0:
                F[i, [1, 2]] = F[i, [2, 1]]
        return F

    @staticmethod
    def _tetrahedron():
        # A tetrahedron: 4 sharp degree-3 corners, 6 crease edges.
        V = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=np.float64)
        F = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
        return V, TestRemeshBotsch._orient_outward(V, F)

    @staticmethod
    def _octahedron():
        # An octahedron: 6 sharp degree-4 corners, 12 crease edges.
        V = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
                     dtype=np.float64)
        F = np.array([[4,0,2],[4,2,1],[4,1,3],[4,3,0],
                      [5,2,0],[5,1,2],[5,3,1],[5,0,3]], dtype=np.int32)
        return V, TestRemeshBotsch._orient_outward(V, F)

    @staticmethod
    def _mesh_volume(V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return float(np.sum(np.einsum('ij,ij->i', np.cross(a, b), c)) / 6.0)

    @staticmethod
    def _face_interior_samples(V, F, k=5):
        # Deterministic barycentric grid over every face (centroid, edge
        # midpoints, interior points). This catches a chamfer triangle whose
        # interior leaves the original surface even when all of its vertices
        # stay on it.
        bary = np.array([(i / k, j / k, (k - i - j) / k)
                         for i in range(k + 1) for j in range(k + 1 - i)])
        pts = (bary[None, :, 0, None] * V[F[:, 0]][:, None, :] +
               bary[None, :, 1, None] * V[F[:, 1]][:, None, :] +
               bary[None, :, 2, None] * V[F[:, 2]][:, None, :])
        return pts.reshape(-1, 3)

    def _assert_coincident(self, v0, f0, u, g, corners, vol_tol=1e-3):
        # (a) every original sharp corner is still present in the output
        for i, c in enumerate(corners):
            d = np.min(np.linalg.norm(u - c, axis=1))
            self.assertLess(d, 1e-6, msg="corner %d %s chamfered (dist %.2e)" % (i, c, d))
        # (b) output face interiors stay on the input surface
        P = self._face_interior_samples(u, g, k=5)
        sqrd = gpytoolbox.squared_distance(P, v0, f0, use_cpp=True)[0]
        self.assertLess(np.max(sqrd), 1e-7,
                        msg="output left the surface (max sq dist %.2e)" % np.max(sqrd))
        # (c) enclosed volume is preserved (chamfering a corner removes volume)
        vin, vout = abs(self._mesh_volume(v0, f0)), abs(self._mesh_volume(u, g))
        self.assertLess(abs(vin - vout) / max(vin, 1e-12), vol_tol,
                        msg="volume changed %.5f -> %.5f" % (vin, vout))

    def test_trapezoidal_prism_no_chamfer(self):
        # The canonical shape that exposed the bug: an extruded trapezoid (a thin
        # slab whose every edge is a sharp crease), built in code so the test
        # needs no external mesh file. After subdividing once and remeshing with
        # auto-detected feature edges it must stay coincident with its input at
        # every iteration count.
        np.random.seed(0)
        v0, f0 = self._extruded_prism([(0,0),(20,0),(25,5),(0,5)], zlo=-1.0, zhi=1.0)
        v0 = gpytoolbox.normalize_points(v0)
        corners = v0.copy()
        v, f = gpytoolbox.subdivide(v0, f0, iters=1)
        for it in [1, 2, 3, 5]:
            u, g = gpytoolbox.remesh_botsch(
                v.copy(), f.copy(), it, h=0.05, detect_feature_edges=True,
                feature_dihedral_threshold=10.0, project=True)
            with self.subTest(iters=it):
                self._assert_coincident(v0, f0, u, g, corners)

    def test_random_points_on_surface(self):
        # The user-facing invariant with (seeded) random sampling: every point
        # sampled on the output lies on the original surface. Uses an in-code
        # trapezoidal prism (no external mesh file).
        np.random.seed(0)
        v0, f0 = self._extruded_prism([(0,0),(20,0),(25,5),(0,5)], zlo=-1.0, zhi=1.0)
        v0 = gpytoolbox.normalize_points(v0)
        v, f = gpytoolbox.subdivide(v0, f0, iters=1)
        u, g = gpytoolbox.remesh_botsch(
            v, f, 2, h=0.05, detect_feature_edges=True,
            feature_dihedral_threshold=10.0, project=True)
        P = gpytoolbox.random_points_on_mesh(u, g, 10000)
        dist = gpytoolbox.squared_distance(P, v0, f0, use_cpp=True)[0]
        self.assertTrue(np.all(dist < 1e-5))

    def test_extruded_prisms_no_chamfer(self):
        # A battery of simple convex prisms: corners preserved and surface kept
        # coincident after auto-detected feature remeshing.
        np.random.seed(0)
        shapes = {
            "square":    [(0,0),(1,0),(1,1),(0,1)],
            "trapezoid": [(0,0),(4,0),(3,1),(1,1)],
            "triangle":  [(0,0),(2,0),(1,1.5)],
            "pentagon":  [(0,0),(2,0),(2.6,1.5),(1,2.6),(-0.6,1.5)],
            "hexagon":   [(1,0),(2,0),(2.5,1),(2,2),(1,2),(0.5,1)],
        }
        for name, poly in shapes.items():
            V, F = self._extruded_prism(poly)
            V = gpytoolbox.normalize_points(V)
            corners = V.copy()
            diag = np.linalg.norm(V.max(0) - V.min(0))
            for it in [3, 8]:
                u, g = gpytoolbox.remesh_botsch(
                    V.copy(), F.copy(), it, h=0.1 * diag,
                    detect_feature_edges=True, feature_dihedral_threshold=20.0,
                    project=True)
                with self.subTest(shape=name, iters=it):
                    self._assert_coincident(V, F, u, g, corners)

    def test_lshape_prism_no_chamfer(self):
        # A non-convex prism (the L has a reflex vertical crease): both convex
        # and reflex degree-3 junctions must survive.
        np.random.seed(0)
        V, F = self._extruded_prism(
            [(0,0),(2,0),(2,1),(1,1),(1,2),(0,2)], zlo=-0.5, zhi=0.5)
        V = gpytoolbox.normalize_points(V)
        corners = V.copy()
        diag = np.linalg.norm(V.max(0) - V.min(0))
        for hf in [0.08, 0.12]:
            for it in [3, 8]:
                u, g = gpytoolbox.remesh_botsch(
                    V.copy(), F.copy(), it, h=hf * diag,
                    detect_feature_edges=True, feature_dihedral_threshold=20.0,
                    project=True)
                with self.subTest(hf=hf, iters=it):
                    self._assert_coincident(V, F, u, g, corners)

    def test_tetrahedron_no_chamfer(self):
        # A closed polyhedron whose every edge is a crease and every vertex a
        # degree-3 junction (different connectivity from a prism).
        np.random.seed(0)
        V, F = self._tetrahedron()
        V = gpytoolbox.normalize_points(V)
        corners = V.copy()
        diag = np.linalg.norm(V.max(0) - V.min(0))
        for hf in [0.08, 0.15, 0.25]:
            for it in [3, 8]:
                u, g = gpytoolbox.remesh_botsch(
                    V.copy(), F.copy(), it, h=hf * diag,
                    detect_feature_edges=True, feature_dihedral_threshold=30.0,
                    project=True)
                with self.subTest(hf=hf, iters=it):
                    self._assert_coincident(V, F, u, g, corners)

    def test_octahedron_no_chamfer(self):
        # Degree-4 junctions: stresses the degree bookkeeping for vertices with
        # more than three incident feature edges.
        np.random.seed(0)
        V, F = self._octahedron()
        V = gpytoolbox.normalize_points(V)
        corners = V.copy()
        diag = np.linalg.norm(V.max(0) - V.min(0))
        for hf in [0.08, 0.15, 0.25]:
            for it in [3, 8]:
                u, g = gpytoolbox.remesh_botsch(
                    V.copy(), F.copy(), it, h=hf * diag,
                    detect_feature_edges=True, feature_dihedral_threshold=30.0,
                    project=True)
                with self.subTest(hf=hf, iters=it):
                    self._assert_coincident(V, F, u, g, corners)

    def test_tilted_prism_no_chamfer(self):
        # A non-axis-aligned prism rules out corners surviving merely because
        # they happen to lie on axis-aligned planes. The coarser target lengths
        # force collapses right at the junctions (which chamfered them before
        # the fix).
        np.random.seed(0)
        V, F = self._extruded_prism(
            [(0,0),(3,0),(2.5,1.2),(0.5,1.0)], zlo=-0.7, zhi=0.7)
        R, _ = np.linalg.qr(np.random.RandomState(0).randn(3, 3))
        V = V @ R.T
        V = gpytoolbox.normalize_points(V)
        corners = V.copy()
        diag = np.linalg.norm(V.max(0) - V.min(0))
        for hf in [0.08, 0.12, 0.18]:
            for it in [5, 8]:
                u, g = gpytoolbox.remesh_botsch(
                    V.copy(), F.copy(), it, h=hf * diag,
                    detect_feature_edges=True, feature_dihedral_threshold=20.0,
                    project=True)
                with self.subTest(hf=hf, iters=it):
                    self._assert_coincident(V, F, u, g, corners)

    def test_feature_junctions_survive_aggressive_collapse(self):
        # Coarsening well below the corner-to-corner spacing forces many
        # collapses around the junctions -- the exact stress that used to
        # chamfer corners. Corners (feature vertices of degree != 2) must never
        # be collapsed away, at any target edge length.
        np.random.seed(0)
        V, F = self._extruded_prism([(0,0),(4,0),(3,1),(1,1)])
        V = gpytoolbox.normalize_points(V)
        corners = V.copy()
        n = 4
        fe = []
        for i in range(n):                  # all 12 prism edges are creases
            j = (i + 1) % n
            fe += [[i, j], [n + i, n + j], [i, n + i]]
        fe = np.array(fe, dtype=np.int32)
        diag = np.linalg.norm(V.max(0) - V.min(0))
        for h in [0.05 * diag, 0.1 * diag, 0.2 * diag, 0.4 * diag]:
            u, g = gpytoolbox.remesh_botsch(
                V.copy(), F.copy(), 10, h=h, feature_edges=fe, project=True)
            with self.subTest(h=h):
                for i, c in enumerate(corners):
                    self.assertLess(np.min(np.linalg.norm(u - c, axis=1)), 1e-6,
                                    msg="corner %d lost at h=%.3f" % (i, h))

    # def test_github_issue_30(self):
    #     np.random.seed(0)
    #     v,f = gpytoolbox.read_mesh("test/unit_tests_data/github_issue_30_input.obj")
    #     ind = gpytoolbox.boundary_vertices(f)
    #     boundary_verts = v[ind,:]
    #     # This used to crash
    #     u,g = gpytoolbox.remesh_botsch(v,f)
    #     gpytoolbox.write_mesh("test/unit_tests_data/github_issue_30_output.obj",u,g)
    #     ind_output = gpytoolbox.boundary_vertices(g)
    #     boundary_verts_output = u[ind_output,:]
    #     # Boundary vertices should not move
    #     for i in range(len(ind)):
    #         dist = np.min(np.linalg.norm(np.tile(boundary_verts[i,:][None,:],(boundary_verts_output.shape[0],1)) - boundary_verts_output,axis=1))
    #         self.assertTrue(dist==0.0)


if __name__ == '__main__':
    unittest.main()
    
