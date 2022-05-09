from context import gpytoolbox
import numpy as np
from scipy.sparse import csr_matrix
import igl
import polyscope as ps

np.random.seed(0)
th = 2*np.pi*np.random.rand(100,1)
P = 2*np.random.rand(100,2) - 1

C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,min_depth=2,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
V,Q = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
bd_children, bd_all = gpytoolbox.quadtree_boundary(CH,A)


# Let's just debug this visually for now
ps.init()
quadtree = ps.register_surface_mesh("test quadtree",V,Q,edge_width=1)
boundary_all = ps.register_point_cloud("children bd",C[bd_children,:])
boundary_all = ps.register_point_cloud("all bd",C[bd_all,:])
ps.show()

