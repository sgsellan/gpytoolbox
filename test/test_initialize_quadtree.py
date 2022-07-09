from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestInitializeQuadtree(unittest.TestCase):
    def test_consistency(self):
        np.random.seed(0)
        for i in range(10):
            th = 2*np.pi*np.random.rand(200,1)
            P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)
            C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=False,max_depth=7,min_depth=4,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
            # Let's check that the arguments are consistents with one another
            for i in range(W.shape[0]):
                # width should be 2/(2^D)
                assert(np.isclose(W[i],2./(2.**(D[i]-1))))
                # children and parents consistent
                for ss in range(4):
                    if CH[i,ss]!=-1:
                        # This is not a leaf node, so it has children
                        # The children's parent is this node
                        assert(PAR[CH[i,ss]]==i)
                # The parent of i must have i as a child
                if PAR[i]>0: #We are not in the supreme dad node
                    assert(i in CH[PAR[i],:])
                # adjacency information
                neighbors_ind = A[:,i].nonzero()[0]
                # Check that they are indeed neighbors... which is equivalent to their Linf distance norm being the half sum of their widths
                for ss in range(len(neighbors_ind)):
                    linf_distance = np.amax(np.abs(C[i] - C[neighbors_ind[ss]]))
                    assert(np.isclose(linf_distance,.5*(W[i] + W[neighbors_ind[ss]])))
                # arguments
                if CH[i,0]==-1:
                    # if this is a child, its depth must be between min and max depth
                    assert(D[i]<=7)
                    assert(D[i]>=4)
            # Test it's graded
            C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=7,min_depth=4,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
            for i in range(W.shape[0]):
                # if we are at a child
                if CH[i,0]==-1:
                    neighbors_ind = A[:,i].nonzero()[0]
                    # Let's find all the child neighbor indeces
                    for ss in range(len(neighbors_ind)):
                        if CH[neighbors_ind[ss],0]==-1:
                            # if it's a child, its depth cannot vary too much
                            assert(np.abs(D[neighbors_ind[ss]]-D[i])<=1)




if __name__ == '__main__':
    unittest.main()



# V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)

# ps.init()
# ps.register_surface_mesh("test quadtree",V,Q,edge_width=1)
# ps.set_navigation_style('planar')
# ps.show()

# 3D...
# P = 2*np.random.rand(20,3)-1
# C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,vmin=np.array([-1,-1,-1]),vmax=np.array([1,1,1]))
# V,Q,H = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
# ps.init()
# ps.register_volume_mesh("octree",V,hexes=H)
# #ps.register_surface_mesh("octree quads",V,faces=Q)
# ps.show()