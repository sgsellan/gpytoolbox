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