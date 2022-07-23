from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import igl
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import polyscope as ps


class TestCompactlySupportedNormal(unittest.TestCase):
    def test_plot(self):
        x = np.reshape(np.linspace(-4,4,1000),(-1,1))
        # v = gpytoolbox.compactly_supported_normal(x, n=4)
        # plt.plot(x,gpytoolbox.compactly_supported_normal(x, n=4,center=np.array([0.5])))
        # plt.plot(x,gpytoolbox.compactly_supported_normal(x, n=3,center=np.array([0.5])))
        # plt.plot(x,gpytoolbox.compactly_supported_normal(x, n=2,center=np.array([0.5])))
        # plt.plot(x,gpytoolbox.compactly_supported_normal(x, n=1,center=np.array([0.5])))
        # plt.show()
        # x, y = np.meshgrid(np.linspace(-1,1,50),np.linspace(-1,1,50))
        # V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
        # print(gpytoolbox.compactly_supported_normal(V[0,:][None,:], n=1,center=np.array([0.5,0.5])))
        # plt.pcolormesh(np.reshape(gpytoolbox.compactly_supported_normal(V, n=1,center=np.array([0.5,0.5])),x.shape))
        # plt.show()


        th = 2*np.pi*np.random.rand(15,1)
        P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)
        C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=6,min_depth=1,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
        V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
        L,stored_at = gpytoolbox.quadtree_fem_laplacian(C,W,CH,D,A)
        #  print(L[10,:].toarray().shape)
        ps.init()
        quadtree = ps.register_surface_mesh("test quadtree",V,Q,edge_width=1)
        
        quadtree.add_scalar_quantity("numeric solve",L.toarray()[25,:],defined_on='faces')
        # quadtree.add_scalar_quantity("groundtruth",gt_fun,defined_on='faces')
        # cloud = ps.register_point_cloud("boundary conditions",stored_at[bb,:])
        ps.show()


if __name__ == '__main__':
    unittest.main()