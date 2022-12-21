from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import polyscope as ps

class TestPoissonSurfaceReconstruction(unittest.TestCase):
    def test_paper_figure(self):
        poly = gpytoolbox.png2poly("test/unit_tests_data/illustrator.png")[0]
        poly = poly - np.min(poly)
        poly = poly/np.max(poly)
        poly = 0.5*poly + 0.25
        poly = 3*poly - 1.5
        num_samples = 40
        np.random.seed(2)
        P, N = gpytoolbox.random_points_on_polyline(poly,num_samples)
        N = - N
        # corner = np.array([-1,-1])
        # h = np.array([0.05,0.05])
        gs = np.array([50,50])
        scalar_mean, scalar_var, grid_vertices = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=0,verbose=True)
        prob_out = 1 - norm.cdf(scalar_mean,0,np.sqrt(scalar_var))
        # # corner = P.min(axis=0)
        # # h = (P.max(axis=0) - P.min(axis=0))/gs
        # # grid_vertices = np.meshgrid(*[np.linspace(corner[dd], corner[dd] + (gs[dd]-1)*h[dd], gs[dd]) for dd in range(P.shape[1])])
        # gx = grid_vertices[0]
        # gy = grid_vertices[1]

        # # Plot mean and variance side by side with colormap
        # fig, ax = plt.subplots(1,3)
        # m0 = ax[0].pcolormesh(gx,gy,np.reshape(scalar_mean,gx.shape), cmap='RdBu',shading='gouraud', vmin=-np.max(np.abs(scalar_mean)), vmax=np.max(np.abs(scalar_mean)))
        # ax[0].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
        # q0 = ax[0].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
        # ax[0].set_title('Mean')
        # divider = make_axes_locatable(ax[0])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(m0, cax=cax, orientation='vertical')
        # # Add colorbar
        # # fig.colorbar(ax[0].imshow(scalar_mean.reshape(gs,order='F')), ax=ax[0])
        # m1 = ax[1].pcolormesh(gx,gy,np.reshape(np.sqrt(scalar_var),gx.shape), cmap='plasma',shading='gouraud')
        # ax[1].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
        # q1 = ax[1].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
        # ax[1].set_title('Variance')
        # divider = make_axes_locatable(ax[1])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(m1, cax=cax, orientation='vertical')

        # m2 = ax[2].pcolormesh(gx,gy,np.reshape(prob_out,gx.shape), cmap='viridis',shading='gouraud')
        # ax[2].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
        # q2 = ax[2].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
        # ax[2].set_title('Variance')
        # divider = make_axes_locatable(ax[2])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(m2, cax=cax, orientation='vertical')
        # # Add colorbar
        # # fig.colorbar(ax[1].imshow(scalar_var.reshape(gs,order='F')), ax=ax[1])
        # plt.show()

    # def test_indicator(self):
    #     np.random.seed(0)
    #     # First test: "uniform" sampling density
    #     # Sample points on a circle
    #     th = 2*np.pi*np.random.rand(60,1)
    #     P = np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
    #     # Normals are the same as positions on a circle
    #     N = np.concatenate((np.cos(th),np.sin(th)),axis=1)

    #     # corner = np.array([-1.5,-1.5])
    #     gs = np.array([80,80])
    #     # h = np.array([0.05,0.05])

    #     scalar_mean, scalar_var = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=1000,verbose=True)

    #     # Plot mean and variance side by side with colormap
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].imshow(scalar_mean.reshape(gs,order='F'))
    #     ax[0].set_title('Mean')
    #     # Add colorbar
    #     fig.colorbar(ax[0].imshow(scalar_mean.reshape(gs,order='F')), ax=ax[0])
    #     ax[1].imshow(scalar_var.reshape(gs,order='F'))
    #     ax[1].set_title('Variance')
    #     # Add colorbar
    #     fig.colorbar(ax[1].imshow(scalar_var.reshape(gs,order='F')), ax=ax[1])
    #     plt.show()
    # def test_3d(self):
    #     v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
    #     print(f.shape)
    #     P = (v[f[:,0],:] + v[f[:,1],:] + v[f[:,2],:])/3.0
    #     N = gpytoolbox.per_face_normals(v,f)
    #     gs = np.array([44,44,44]) #44
    #     # gs = np.array([4,4,3])
    #     scalar_mean, scalar_var = gpytoolbox.poisson_surface_reconstruction(P,N,corner=np.array([-1.1,-1.1,-1.1]),h=np.array([0.05,0.05,0.05]),gs=gs,solve_subspace_dim=3000,stochastic=True,verbose=True)
    #     tet_verts, tets = gpytoolbox.regular_cube_mesh(gs[0],type='hex')
    #     tet_verts = 2.2*tet_verts - 1.1
    #     R = np.array([[0.0,0.0,1.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]]) @ np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,-1.0,0.0]])
    #     tet_verts = tet_verts @ R
    #     tet_verts[:,0] = - tet_verts[:,0]
    #     tet_verts[:,1] = - tet_verts[:,1]
    #     ps.init()
    #     ps_vol = ps.register_volume_mesh("test volume mesh", tet_verts, hexes=tets, enabled=False)
    #     ps_vol.add_scalar_quantity("mean", scalar_mean)
    #     ps_vol.add_scalar_quantity("sigma", scalar_var)
    #     sample_points = ps.register_point_cloud("sample points", P)
    #     sample_points.add_vector_quantity("sample normals", N, enabled=True)
    #     ps.show()


if __name__ == '__main__':
    unittest.main()