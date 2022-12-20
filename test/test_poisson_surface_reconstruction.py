from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import matplotlib.pyplot as plt
import polyscope as ps

class TestPoissonSurfaceReconstruction(unittest.TestCase):
    # def test_indicator(self):
    #     np.random.seed(0)
    #     # First test: "uniform" sampling density
    #     # Sample points on a circle
    #     th = 2*np.pi*np.random.rand(80,1)
    #     P = np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
    #     # Normals are the same as positions on a circle
    #     N = np.concatenate((np.cos(th),np.sin(th)),axis=1)

    #     # corner = np.array([-1.5,-1.5])
    #     gs = np.array([80,80])
    #     # h = np.array([0.05,0.05])

    #     scalar_mean, scalar_var = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=1000)

        # Plot mean and variance side by side with colormap
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(scalar_mean.reshape(gs,order='F'))
        # ax[0].set_title('Mean')
        # # Add colorbar
        # fig.colorbar(ax[0].imshow(scalar_mean.reshape(gs,order='F')), ax=ax[0])
        # ax[1].imshow(scalar_var.reshape(gs,order='F'))
        # ax[1].set_title('Variance')
        # # Add colorbar
        # fig.colorbar(ax[1].imshow(scalar_var.reshape(gs,order='F')), ax=ax[1])
        # plt.show()
    def test_3d(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        P = (v[f[:,0],:] + v[f[:,1],:] + v[f[:,2],:])/3.0
        N = gpytoolbox.per_face_normals(v,f)
        gs = np.array([44,44,44])
        scalar_mean, scalar_var = gpytoolbox.poisson_surface_reconstruction(P,N,corner=np.array([-1.1,-1.1,-1.1]),h=np.array([0.05,0.05,0.05]),gs=gs,solve_subspace_dim=3000,stochastic=True)
        tet_verts, tets = gpytoolbox.regular_cube_mesh(gs[0],type='hex')
        tet_verts = 2.2*tet_verts - 1.1
        R = np.array([[0.0,0.0,1.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]]) @ np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,-1.0,0.0]])
        tet_verts = tet_verts @ R
        tet_verts[:,0] = - tet_verts[:,0]
        tet_verts[:,1] = - tet_verts[:,1]
        ps.init()
        ps_vol = ps.register_volume_mesh("test volume mesh", tet_verts, hexes=tets, enabled=False)
        ps_vol.add_scalar_quantity("mean", scalar_mean)
        ps_vol.add_scalar_quantity("sigma", scalar_var)
        sample_points = ps.register_point_cloud("sample points", P)
        sample_points.add_vector_quantity("sample normals", N, enabled=True)
        ps.show()


if __name__ == '__main__':
    unittest.main()