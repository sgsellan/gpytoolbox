from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestGridNeighbors(unittest.TestCase):
    def test_2d(self):
        # Anisotropic grid numbers
        gs = np.array([90,85])
        # This should be *only* the 8 neighbors at a distance of h
        h = np.array([1.,1.])
        corner = np.array([0.,0.])
        dim = 2
        grid_vertices = np.meshgrid(*[np.linspace(corner[dd], corner[dd] + (gs[dd]-1)*h[dd], gs[dd]) for dd in range(dim)])
        grid_vertices = np.array(grid_vertices).reshape(dim, -1).T
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=1)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 1.0
            # All the detected neighbors should have distance 1
            self.assertTrue(np.all(dist == 1.0))
            # And there should be 4 of them
            self.assertTrue(neighbors.shape[0] == 4)

        # Now let's include diagonals
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=True, include_self=False,order=1,output_unique=True)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 1.0
            # All the detected neighbors should have distance between 1 and sqrt(2)
            self.assertTrue(np.all(dist >= 1.0))
            self.assertTrue(np.all(dist <= np.sqrt(2)))
            # There should be eight unique ones
            self.assertTrue(neighbors.shape[0] == 8)
        
        # Let's try with include_self
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=True, include_self=True,order=1)
        should_be_self = N[0,:]
        self_ind = np.arange(N.shape[1])
        self.assertTrue(np.all(should_be_self == self_ind))
        # Without diagonals
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=True,order=1)
        should_be_self = N[0,:]
        self_ind = np.arange(N.shape[1])
        self.assertTrue(np.all(should_be_self == self_ind))
        # print(should_be_self)

    def test_3d(self):
        gs = np.array([12,31,25])
        # This should be *only* the 8 neighbors at a distance of h
        h = np.array([1.,1.,1.])
        corner = np.array([0.,0.,0.])
        dim = 3
        grid_vertices = np.meshgrid(*[np.linspace(corner[dd], corner[dd] + (gs[dd]-1)*h[dd], gs[dd]) for dd in range(dim)], indexing='ij')
        grid_vertices = np.array(grid_vertices).reshape(dim, -1,order='F').T
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=1)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 1.0
            # All the detected neighbors should have distance 1
            self.assertTrue(np.all(dist == 1.0))
            # And there should be 6 of them
            self.assertTrue(neighbors.shape[0] == 6)

        # Now let's include diagonals
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=True, include_self=False,order=1,output_unique=True)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 1.0
            # All the detected neighbors should have distance between 1 and sqrt(3)
            self.assertTrue(np.all(dist >= 1.0))
            self.assertTrue(np.all(dist <= np.sqrt(3)))
            # There should be 26 unique ones
            self.assertTrue(neighbors.shape[0] == 26)

    def test_order_two(self):
        gs = np.array([90,85])
        # This should be *only* the 8 neighbors at a distance of h
        h = np.array([1.,1.])
        corner = np.array([0.,0.])
        dim = 2
        grid_vertices = np.meshgrid(*[np.linspace(corner[dd], corner[dd] + (gs[dd]-1)*h[dd], gs[dd]) for dd in range(dim)])
        grid_vertices = np.array(grid_vertices).reshape(dim, -1).T
        # N_debug = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=1)
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=2,output_unique=True)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 0.0
            # All the detected neighbors should have distance between 0 and 2
            self.assertTrue(np.all(dist >= 0.0))
            eps = 0.000001
            self.assertTrue(np.all(dist <= 2.0+eps))
            # Aand there should be 13 of them
            self.assertTrue(N.shape[0] == 13)

        # Now let's include diagonals
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=True, include_self=False,order=2,output_unique=True)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 0.0
            # All the detected neighbors should have distance between 0 and 2
            self.assertTrue(np.all(dist >= 0.0))
            eps = 0.000001
            self.assertTrue(np.all(dist <= np.sqrt(8)+eps))
            # And there should be 25 of them
            self.assertTrue(N.shape[0] == 25)

    def test_order_two_3d(self):
        gs = np.array([12,31,25])
        # This should be *only* the 8 neighbors at a distance of h
        h = np.array([1.,1.,1.])
        corner = np.array([0.,0.,0.])
        dim = 3
        grid_vertices = np.meshgrid(*[np.linspace(corner[dd], corner[dd] + (gs[dd]-1)*h[dd], gs[dd]) for dd in range(dim)], indexing='ij')
        grid_vertices = np.array(grid_vertices).reshape(dim, -1,order='F').T
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=1)
        # N_debug = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=1)
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=2,output_unique=True)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 0.0
            # All the detected neighbors should have distance between 0 and 2
            self.assertTrue(np.all(dist >= 0.0))
            eps = 0.000001
            self.assertTrue(np.all(dist <= 2.0+eps))
            # Aand there should be 13 of them
            # self.assertTrue(N.shape[0] == 13)

        # Now let's include diagonals
        N = gpytoolbox.grid_neighbors(gs, include_diagonals=True, include_self=False,order=2,output_unique=True)
        for i in range(N.shape[1]):
            grid_vert = grid_vertices[i,:]
            neighbors = grid_vertices[N[:,i],:]
            grid_vert_tiled = np.tile(grid_vert, (neighbors.shape[0],1))
            dist = np.linalg.norm(grid_vert_tiled - neighbors, axis=1)
            dist[N[:,i]<0] = 0.0
            # All the detected neighbors should have distance between 0 and 2
            self.assertTrue(np.all(dist >= 0.0))
            eps = 0.000001
            self.assertTrue(np.all(dist <= 2*np.sqrt(3)+eps))
            # And there should be 125 of them
            self.assertTrue(N.shape[0] == 125)


if __name__ == '__main__':
    unittest.main()

