from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestSubdivideQuad(unittest.TestCase):
    def test_consistency(self):
        np.random.seed(0)
        for i in range(10):
            th = 2*np.pi*np.random.rand(200,1)
            P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)
            C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=False,max_depth=7,min_depth=4,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
            leaf_ind = gpytoolbox.quadtree_children(CH)
            # Let's choose an index:
            ind = leaf_ind[20]
            self.assertTrue(CH[ind,0]==-1)
            C1,W1,CH1,PAR1,D1,A1 = gpytoolbox.subdivide_quad(ind,C,W,CH,PAR,D,A,graded=False)
            # No longer a leaf node
            self.assertTrue(CH1[ind,0]>=0)
            # In fact, it should point to the final four indices
            num_cells = C.shape[0]
            true_children = np.array([num_cells,num_cells+1,num_cells+2,num_cells+3])
            self.assertTrue((CH1[ind,:]==true_children).all())
            # Widths are halved
            self.assertTrue(W1[num_cells]==(W[ind]/2.))
            self.assertTrue(W1[num_cells+1]==(W[ind]/2.))
            self.assertTrue(W1[num_cells+2]==(W[ind]/2.))
            self.assertTrue(W1[num_cells+3]==(W[ind]/2.))
            # Depths are one plus
            self.assertTrue(D1[num_cells]==(D[ind]+1))
            self.assertTrue(D1[num_cells+1]==(D[ind]+1))
            self.assertTrue(D1[num_cells+2]==(D[ind]+1))
            self.assertTrue(D1[num_cells+3]==(D[ind]+1))
            # All are children
            self.assertTrue((CH1[num_cells,:]==-1).all())
            self.assertTrue((CH1[num_cells+1,:]==-1).all())
            self.assertTrue((CH1[num_cells+2,:]==-1).all())
            self.assertTrue((CH1[num_cells+3,:]==-1).all())
            # Parent is ind
            self.assertTrue(PAR1[num_cells]==ind)
            self.assertTrue(PAR1[num_cells+1]==ind)
            self.assertTrue(PAR1[num_cells+2]==ind)
            self.assertTrue(PAR1[num_cells+3]==ind)
            





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