import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestEdgeEdgeDistance(unittest.TestCase):
    # This isn't too complex, probably could use being expanded
    def test_synthetic(self):
        # Build a polyline; for example, a square
        P1 = np.array([0.0,0.0,0.0])
        P2 = np.array([1.0,0.0,0.0])
        Q1 = np.array([0.0,1.0,0.0])
        Q2 = np.array([1.0,1.0,0.0])
        dist,R1,R2 = gpytoolbox.edge_edge_distance(P1,Q1,P2,Q2)
        # print(dist,R1,R2)
        self.assertTrue(np.isclose(dist,1.0))
        self.assertTrue(np.isclose(R1,np.array([0.0,0.0,0.0])).all())
        self.assertTrue(np.isclose(R2,np.array([1.0,0.0,0.0])).all())
    def test_consistency_meshes(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
        np.random.seed(0)
        for mesh in meshes:
            v,f = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
            for i in range(100): 
                # Now pick a random edge
                E = gpytoolbox.edges(f)
                # Set random seed
                e1 = np.random.randint(E.shape[0])
                P1 = v[E[e1,0],:]
                Q1 = v[E[e1,1],:]
                # Now pick a random edge
                e2 = np.random.randint(E.shape[0])
                P2 = v[E[e2,0],:]
                Q2 = v[E[e2,1],:]
                dist,R1,R2 = gpytoolbox.edge_edge_distance(P1,Q1,P2,Q2)
                self.is_correct_distance(P1,Q1,P2,Q2,np.linalg.norm(R1-R2))
                self.is_correct_distance(P1,Q1,P2,Q2,dist)
                # This is redundant but just in case
                self.assertTrue(np.isclose(dist,np.linalg.norm(R1-R2)))
    
    def is_correct_distance(self,P1,Q1,P2,Q2,dist):
        # Sample the first edge densely
        points_1 = np.linspace(P1,Q1,100)
        # Sample the second edge densely
        points_2 = np.linspace(P2,Q2,100)
        # Minimum distance between the rows of points_1 and points_2
        min_dist = np.min(np.linalg.norm(points_1[:,np.newaxis,:]-points_2[np.newaxis,:,:],axis=2))
        # print("min_dist :",min_dist)
        # print("dist :",dist)
        self.assertTrue(np.isclose(dist,min_dist))





        

if __name__ == '__main__':
    unittest.main()
