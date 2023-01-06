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