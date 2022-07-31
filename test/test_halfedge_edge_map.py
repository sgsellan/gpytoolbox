from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestHalfedgeEdgeMap(unittest.TestCase):

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        he_m, E_m, he_to_E_m, E_to_he_m = gpy.halfedge_edge_map(f, assume_manifold=True)
        he_nm, E_nm, he_to_E_nm, E_to_he_nm = gpy.halfedge_edge_map(f, assume_manifold=False)
        self.assertTrue(np.all(E_m == E_nm))
        self.assertTrue(np.all(he_m == he_nm))
        self.assertTrue(np.all(he_m == gpy.halfedges(f)))
        self.E_to_he_equality(E_m, E_nm)
        self.variety_of_asserts(he_m, E_m, he_to_E_m, E_to_he_m, True)
        self.variety_of_asserts(he_nm, E_nm, he_to_E_nm, E_to_he_nm, False)

        #This mesh is boundary-only
        self.assertTrue(np.all(E_to_he_m[:,1,:] == -1))
        self.assertTrue(all([ethe.shape[0]==1 for ethe in E_to_he_nm]))

    def test_two_triangles(self):
        f = np.array([[0,1,2],[2,1,3]],dtype=int)
        he_m, E_m, he_to_E_m, E_to_he_m = gpy.halfedge_edge_map(f, assume_manifold=True)
        he_nm, E_nm, he_to_E_nm, E_to_he_nm = gpy.halfedge_edge_map(f, assume_manifold=False)
        self.assertTrue(np.all(E_m == E_nm))
        self.assertTrue(np.all(he_m == he_nm))
        self.assertTrue(np.all(he_m == gpy.halfedges(f)))
        self.E_to_he_equality(E_m, E_nm)
        self.variety_of_asserts(he_m, E_m, he_to_E_m, E_to_he_m, True)
        self.variety_of_asserts(he_nm, E_nm, he_to_E_nm, E_to_he_nm, False)
    
    def test_variety_of_meshes(self):
        meshes = ["airplane.obj", "armadillo.obj", "bunny.obj", "bunny_oded.obj", "mountain.obj", "wooden-chair-remesher-bug.obj"]
        for mesh in meshes:
            v,f = gpy.read_mesh("test/unit_tests_data/" + mesh)

            do_m = True
            do_nm = v.shape[0]<5000

            if do_nm:
                he_nm, E_nm, he_to_E_nm, E_to_he_nm = gpy.halfedge_edge_map(f, assume_manifold=False)
                self.assertTrue(np.all(he_nm == gpy.halfedges(f)))
                self.variety_of_asserts(he_nm, E_nm, he_to_E_nm, E_to_he_nm, False)
            if do_m:
                he_m, E_m, he_to_E_m, E_to_he_m = gpy.halfedge_edge_map(f, assume_manifold=True)
                self.assertTrue(np.all(he_m == gpy.halfedges(f)))
                self.variety_of_asserts(he_m, E_m, he_to_E_m, E_to_he_m, True)
                if do_nm:
                    self.assertTrue(np.all(E_m == E_nm))
                    self.assertTrue(np.all(he_m == he_nm))
                    self.E_to_he_equality(E_m, E_nm)

    def E_to_he_equality(self, E_to_he_manifold, E_to_he_nonmanifold):
        self.assertTrue(E_to_he_manifold.shape[0] == len(E_to_he_nonmanifold))
        for e in range(len(E_to_he_nonmanifold)):
            valid_indices = (E_to_he_manifold[e] >= 0)
            self.assertTrue(np.all(E_to_he_nonmanifold[e] == E_to_he_manifold[e][valid_indices]))

    def variety_of_asserts(self, he, E, he_to_E, E_to_he, assume_manifold):
        # Make sure everything is correct in he_to_E
        for j in range(3):
            self.assertTrue(np.all(np.sort(E[he_to_E[:,j],:], axis=-1)==np.sort(he[:,j,:], axis=-1)))
        # Make sure everything is correct in E_to_he
        if assume_manifold:
            for e in range(E_to_he.shape[0]):
                he_ind = (E_to_he[e][:,0], E_to_he[e][:,1])
                he_ind_valid = (he_ind[0][he_ind[0]>=0], he_ind[1][he_ind[1]>=0])
                self.assertTrue(np.all(np.sort(he[he_ind_valid], axis=-1) == np.sort(E[e,:], axis=-1)))
        else:
            self.assertTrue(all([np.all(np.sort(he[E_to_he[e][:,0],E_to_he[e][:,1],:], axis=-1) == np.sort(E[e,:], axis=-1)) for e in range(len(E_to_he))]))
        # Test reversibility
        for i in range(he_to_E.shape[0]):
            for j in range(3):
                e = he_to_E[i,j]
                if (E_to_he[e][0,:]==np.array([i,j])).all():
                    self.assertTrue(E_to_he[e].shape==(1,2) or
                        (E_to_he[e][1,:]==np.array([-1,-1])).all() or
                        he_to_E[E_to_he[e][1,0],E_to_he[e][1,1]]==e)
                else:
                    self.assertTrue((E_to_he[e][1,:]==np.array([i,j])).all())
                    self.assertTrue((E_to_he[e][0,:]==np.array([-1,-1])).all() or
                        he_to_E[E_to_he[e][0,0],E_to_he[e][0,1]]==e)

if __name__ == '__main__':
    unittest.main()

