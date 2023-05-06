import numpy as np
from .context import gpytoolbox
from .context import unittest
import matplotlib.pyplot as plt

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
        # print(np.min(verts_S))
        self.assertTrue(np.allclose(verts_S, 0.0))
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
        filename = "test/unit_tests_data/illustrator.png"
        poly = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        poly = gpytoolbox.normalize_points(poly)
        # plt.plot(poly[:,0],poly[:,1])
        # plt.show()
        E = gpytoolbox.edge_indices(poly.shape[0],closed=True)
        S = gpytoolbox.signed_distance(GV, poly, E)[0]
        # S = np.linalg.norm(GV, axis=1)-0.5
        verts, edge_list = gpytoolbox.marching_squares(S, GV, n+1, n+1)
        # # All verts should have zero sdf
        verts_S = gpytoolbox.signed_distance(verts, poly, E)[0]**2.0
        
        
        plt.pcolormesh(gx.reshape(n+1,n+1),gy.reshape(n+1,n+1),S.reshape(n+1,n+1))
        plt.plot(poly[:,0],poly[:,1])
        # for i in range(edge_list.shape[0]):
        #     plt.plot([verts[edge_list[i,0],0],verts[edge_list[i,1],0]],
        #              [verts[edge_list[i,0],1],verts[edge_list[i,1],1]],
        #              'k-')
        plt.show()

        self.assertTrue(np.allclose(verts_S, 0.0,atol=1e-4))
            




if __name__ == '__main__':
    unittest.main()
