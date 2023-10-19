from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestNonManifoldEdges(unittest.TestCase):
    # There is not much to test here that goes beyond just inputting the
    # definition of the function, but we can make sure that a few conditions
    # are fulfilled.

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        ne = gpy.non_manifold_edges(f)
        # no non-manifold edges
        self.assertTrue(len(ne)==0)

    def test_simple_nonmanifold(self):
        f = np.array([[0,1,2],[0,2,3],[2,0,4]],dtype=int)
        ne = gpy.non_manifold_edges(f)
        ne_gt = np.array([[0,2]],dtype=int)
        self.assertTrue(np.all(ne==ne_gt))
    
    def test_bunny(self):
        _,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        num_faces = f.shape[0]
        he = gpy.halfedges(f).reshape(-1,2)

        for it in range(100):
            # pick a random edge in he
            i = np.random.randint(he.shape[0])
            random_edge = he[i,:]
            # now we add a new face that contains this edge
            new_face = np.array([random_edge[0],random_edge[1],num_faces],dtype=int)
            # insert this face into a random position in f
            f_bad = np.insert(f,np.random.randint(f.shape[0]),new_face,axis=0)
            # are there any non-manifold edges?
            ne = gpy.non_manifold_edges(f_bad)
            self.assertTrue(ne.shape[0]==1)
            # sort random_edge
            random_edge = np.sort(random_edge)
            # check that ne is random edge
            self.assertTrue(np.all(ne==random_edge))
            # print(random_edge)
        
        # now let's add them sequentially
        f = f_bad.copy()
        ne_gt = ne.copy()
        rng = np.random.default_rng(5)
        for it in range(10):
            # pick a random edge in he
            i = rng.integers(he.shape[0])
            random_edge = he[i,:]
            # now we add a new face that contains this edge
            new_face = np.array([random_edge[0],random_edge[1],num_faces+it],dtype=int)
            # insert this face into a random position in f
            f = np.insert(f,np.random.randint(f.shape[0]),new_face,axis=0)
            # are there any non-manifold edges?
            ne = gpy.non_manifold_edges(f)
            
            self.assertTrue(ne.shape[0]==it+2)
            # sort random_edge
            random_edge = np.sort(random_edge)
            ne_gt = np.vstack((ne_gt,random_edge))
            # sort ne_gt lexicographically
            ne_gt = np.unique(ne_gt,axis=0,)
            # should match
            self.assertTrue(np.all(ne==ne_gt))

        


if __name__ == '__main__':
    unittest.main()