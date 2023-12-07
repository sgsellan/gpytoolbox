import numpy as np
from .context import gpytoolbox
from .context import unittest
# import matplotlib.pyplot as plt

class TestMarchingSquares(unittest.TestCase):
    # This isn't too complex, probably could use being expanded
    def test_circle(self):
        # Build a polyline; for example, a square
        n = 200
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), indexing='ij')
        GV = np.vstack((gx.flatten(), gy.flatten())).T
        S = np.linalg.norm(GV, axis=1)-0.5
        verts, edge_list = gpytoolbox.marching_squares(S, GV, n+1, n+1)
        # All verts should have zero sdf
        verts_S = (np.linalg.norm(verts, axis=1)-0.5)**2.0
        #Normals point out
        R = 0.5*(verts[edge_list[:,1],:] + verts[edge_list[:,0],:])
        N = (verts[edge_list[:,1],:] - verts[edge_list[:,0],:]) @ np.array([[0., -1.], [1., 0.]])
        self.assertTrue(np.all(np.sum(R*N, axis=-1)>0.))
        self.assertTrue(np.allclose(verts_S, 0.0))
        # for i in range(edge_list.shape[0]):
        #     plt.plot([verts[edge_list[i,0],0],verts[edge_list[i,1],0]],
        #              [verts[edge_list[i,0],1],verts[edge_list[i,1],1]],
        #              'k-')
        #     plt.quiver(0.5*(verts[edge_list[i,0],0]+verts[edge_list[i,1],0]),
        #         0.5*(verts[edge_list[i,0],1]+verts[edge_list[i,1],1]),
        #         N[i,0], N[i,1])
        # plt.show()
    def test_png(self):
        # Build a polyline; for example, a square
        n = 100
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        GV = np.vstack((gx.flatten(), gy.flatten())).T
        filename = "test/unit_tests_data/illustrator.png"
        poly = gpytoolbox.png2poly(filename)[0]
        poly = poly[::10,:]
        poly = gpytoolbox.normalize_points(poly)
        # plt.plot(poly[:,0],poly[:,1])
        # plt.show()
        E = gpytoolbox.edge_indices(poly.shape[0],closed=True)
        S = gpytoolbox.signed_distance(GV, poly, E)[0]
        # S = np.linalg.norm(GV, axis=1)-0.5
        verts, edge_list = gpytoolbox.marching_squares(S, GV, n+1, n+1)
        # # All verts should have zero sdf
        verts_S = gpytoolbox.signed_distance(verts, poly, E)[0]**2.0
        self.assertTrue(np.allclose(verts_S, 0.0,atol=1e-4))
    def test_square(self):
        # Build a polyline; for example, a square
        n = 21
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        GV = np.vstack((gx.flatten(), gy.flatten())).T
        poly = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) # counter clockwise
        poly = gpytoolbox.normalize_points(poly)
        # plt.plot(poly[:,0],poly[:,1])
        # plt.show()
        E = gpytoolbox.edge_indices(poly.shape[0],closed=True)
        S = gpytoolbox.signed_distance(GV, poly, E)[0]
        # S = np.linalg.norm(GV, axis=1)-0.5
        verts, edge_list = gpytoolbox.marching_squares(S, GV, n+1, n+1)
        # # All verts should have zero sdf
        verts_S = gpytoolbox.signed_distance(verts, poly, E)[0]**2.0        
        #Normals point out
        R = 0.5*(verts[edge_list[:,1],:] + verts[edge_list[:,0],:])
        N = (verts[edge_list[:,1],:] - verts[edge_list[:,0],:]) @ np.array([[0., -1.], [1., 0.]])
        self.assertTrue(np.all(np.sum(R*N, axis=-1)>0.))
        self.assertTrue(np.allclose(verts_S, 0.0,atol=1e-4))

    def test_signs(self):
        # Build a polyline; for example, a square
        n = 250
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        GV = np.vstack((gx.flatten(), gy.flatten())).T
        filename = "test/unit_tests_data/illustrator.png"
        poly = gpytoolbox.png2poly(filename)[0]
        poly = poly[::10,:]
        poly = gpytoolbox.normalize_points(poly)
        E = gpytoolbox.edge_indices(poly.shape[0],closed=True)
        S1 = gpytoolbox.signed_distance(GV, poly, E)[0]
        # S = np.linalg.norm(GV, axis=1)-0.5
        verts, edge_list = gpytoolbox.marching_squares(S1, GV, n+1, n+1)
        S2 = gpytoolbox.signed_distance(GV, verts, edge_list)[0]
        self.assertTrue(np.allclose(S1,S2,atol=1e-2))

    def test_empty(self):
        n = 250
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        GV = np.vstack((gx.flatten(), gy.flatten())).T
        S = np.ones((GV.shape[0],1))
        # this used to cause a crash
        verts, edge_list = gpytoolbox.marching_squares(S, GV, n+1, n+1)
        self.assertTrue(verts.shape[0]==0)



if __name__ == '__main__':
    unittest.main()
