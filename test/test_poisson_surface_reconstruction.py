from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
from scipy.stats import norm
import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import polyscope as ps

class TestPoissonSurfaceReconstruction(unittest.TestCase):
    def test_paper_figure(self):
        poly = gpytoolbox.png2poly("test/unit_tests_data/illustrator.png")[0]
        poly = poly - np.min(poly)
        poly = poly/np.max(poly)
        poly = 0.5*poly + 0.25
        V = 3*poly - 1.5
        V = V[0:V.shape[0]-1,:] # remove the last point, which is a duplicate of the first
        # unique_ind = np.unique(V, axis=0, return_index=True)[1]
        # V = V[unique_ind,:]
        n = 40
        np.random.seed(2)
        EC = gpytoolbox.edge_indices(V.shape[0],closed=False)
        P,I,_ = gpytoolbox.random_points_on_mesh(V, EC, n, return_indices=True)
        vecs = V[EC[:,0],:] - V[EC[:,1],:]
        vecs /= np.linalg.norm(vecs, axis=1)[:,None]
        J = np.array([[0., -1.], [1., 0.]])
        N = vecs @ J.T
        N = N[I,:]
        gs = np.array([50,50])
        # corner = np.array([-1,-1])
        # h = np.array([0.04,0.04])
        scalar_mean, scalar_var, grid_vertices = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=0,verbose=False,stochastic=True)
        # corner = P.min(axis=0)
        # h = (P.max(axis=0) - P.min(axis=0))/gs

        prob_out = 1 - norm.cdf(scalar_mean,0,np.sqrt(scalar_var))

        # The first test we can run is generate many points inside the shape
        # and make sure that the probability of being outside is low
        n = 1000
        candidate_points = np.random.uniform(-1,1,size=(n,2))
        distances = gpytoolbox.signed_distance_polygon(candidate_points, V)
        # print(distances)
        points_inside = candidate_points[distances < 0,:]
        points_outside = candidate_points[distances > 0,:]
        grid_vertices = np.array(grid_vertices).reshape(2, -1).T
        corner = grid_vertices.min(axis=0)
        h = (grid_vertices.max(axis=0) - grid_vertices.min(axis=0))/gs
        W = gpytoolbox.fd_interpolate(candidate_points,gs,h,corner=corner)
        candidate_mean = W @ scalar_mean
        candidate_var = W @ scalar_var
        candidate_prob_out = 1 - norm.cdf(candidate_mean,0,np.sqrt(candidate_var))

        correctly_labeled_inside = np.sum(candidate_mean[distances < 0] < 0) / points_inside.shape[0]
        # It should correctly label most of the points inside
        self.assertTrue(correctly_labeled_inside > 0.75)
        # and outside
        correctly_labeled_outside = np.sum(candidate_mean[distances > 0] > 0) / points_outside.shape[0]
        self.assertTrue(correctly_labeled_outside > 0.75)

        # The same should happen with prob_out
        correctly_labeled_inside = np.sum(candidate_prob_out[distances < 0] > 0.5) / points_inside.shape[0]
        # It should correctly label most of the points inside
        # print(correctly_labeled_inside)
        self.assertTrue(correctly_labeled_inside > 0.75)
        # and outside
        correctly_labeled_outside = np.sum(candidate_prob_out[distances > 0] < 0.5) / points_outside.shape[0]
        # print(correctly_labeled_outside)
        self.assertTrue(correctly_labeled_outside > 0.75)

        # Where is the highest variance?
        variance_argsort = np.argsort(scalar_var)
        # The lowest variance should be near data points
        for i in range(20):
            distance_to_closest_data_point = np.min(np.linalg.norm(grid_vertices[variance_argsort[i],:] - P,axis=1))
            self.assertTrue(distance_to_closest_data_point < 0.1)

    def test_indicator(self):
        np.random.seed(0)
        # Sample points on a circle
        th = 2*np.pi*np.random.rand(200,1)
        P = 0.75*np.concatenate((np.cos(th)+0.1,np.sin(th)-0.1),axis=1)
        V = P
        # Normals are the same as positions on a circle
        N = np.concatenate((np.cos(th),np.sin(th)),axis=1)
        gs = np.array([100,100])
        scalar_mean, scalar_var, grid_vertices = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=1000,verbose=False,stochastic=True)
        prob_out = 1 - norm.cdf(scalar_mean,0,np.sqrt(scalar_var))

        # The first test we can run is generate many points inside the shape
        # and make sure that the probability of being outside is low
        n = 1000
        candidate_points = np.random.uniform(-1,1,size=(n,2))
        distances = gpytoolbox.signed_distance_polygon(candidate_points, V)
        # print(distances)
        points_inside = candidate_points[distances < 0,:]
        points_outside = candidate_points[distances > 0,:]
        grid_vertices = np.array(grid_vertices).reshape(2, -1).T
        corner = grid_vertices.min(axis=0)
        h = (grid_vertices.max(axis=0) - grid_vertices.min(axis=0))/gs
        W = gpytoolbox.fd_interpolate(candidate_points,gs,h,corner=corner)
        candidate_mean = W @ scalar_mean
        candidate_var = W @ scalar_var
        candidate_prob_out = 1 - norm.cdf(candidate_mean,0,np.sqrt(candidate_var))

        # plt.scatter(candidate_points[:,0],candidate_points[:,1],c=candidate_mean)
        # plt.show()

        correctly_labeled_inside = np.sum(candidate_mean[distances < 0] < 0) / points_inside.shape[0]
        # It should correctly label most of the points inside
        self.assertTrue(correctly_labeled_inside > 0.75)
        # and outside
        correctly_labeled_outside = np.sum(candidate_mean[distances > 0] > 0) / points_outside.shape[0]
        self.assertTrue(correctly_labeled_outside > 0.75)

        # The same should happen with prob_out
        correctly_labeled_inside = np.sum(candidate_prob_out[distances < 0] > 0.5) / points_inside.shape[0]
        # It should correctly label most of the points inside
        # print(correctly_labeled_inside)
        self.assertTrue(correctly_labeled_inside > 0.75)
        # and outside
        correctly_labeled_outside = np.sum(candidate_prob_out[distances > 0] < 0.5) / points_outside.shape[0]
        # print(correctly_labeled_outside)
        self.assertTrue(correctly_labeled_outside > 0.75)

        # Where is the highest variance?
        variance_argsort = np.argsort(scalar_var)
        # The lowest variance should be near data points
        for i in range(20):
            distance_to_closest_data_point = np.min(np.linalg.norm(grid_vertices[variance_argsort[i],:] - P,axis=1))
            self.assertTrue(distance_to_closest_data_point < 0.1)
    def test_3d(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # print(f.shape)
        P = (v[f[:,0],:] + v[f[:,1],:] + v[f[:,2],:])/3.0
        N = gpytoolbox.per_face_normals(v,f)
        gs = np.array([44,44,44]) #44
        # gs = np.array([10,11,15])
        
        if os.name == 'nt':
            # Windows machine in github action can't handle this test
            pass
        else:
            scalar_mean, scalar_var, grid_vertices = gpytoolbox.poisson_surface_reconstruction(P,N,corner=np.array([-1.1,-1.1,-1.1]),h=np.array([0.05,0.05,0.05]),gs=gs,solve_subspace_dim=3000,stochastic=True,verbose=False)
            grid_vertices = np.array(grid_vertices).reshape(3, -1,order='F').T
            # Where is the highest variance?
            variance_argsort = np.argsort(scalar_var)
            # The lowest variance should be near data points
            for i in range(20):
                distance_to_closest_data_point = np.min(np.linalg.norm(grid_vertices[variance_argsort[i],:] - P,axis=1))
            # self.assertTrue(distance_to_closest_data_point < 0.3)
        
        # Once we have proper in-out segmentation, we should expand this test


        # tet_verts, tets = gpytoolbox.regular_cube_mesh(gs[0],type='hex')
        # tet_verts = 2.2*tet_verts - 1.1
        # R = np.array([[0.0,0.0,1.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]]) @ np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,-1.0,0.0]])
        # tet_verts = tet_verts @ R
        # tet_verts[:,0] = - tet_verts[:,0]
        # tet_verts[:,1] = - tet_verts[:,1]
        # ps.init()
        # ps_vol = ps.register_volume_mesh("test volume mesh", tet_verts, hexes=tets, enabled=False)
        # ps_vol.add_scalar_quantity("mean", scalar_mean)
        # ps_vol.add_scalar_quantity("sigma", scalar_var)
        # sample_points = ps.register_point_cloud("sample points", P)
        # sample_points.add_vector_quantity("sample normals", N, enabled=True)
        # ps.show()


if __name__ == '__main__':
    unittest.main()