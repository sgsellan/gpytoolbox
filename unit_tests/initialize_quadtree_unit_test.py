from context import gpytoolbox
import numpy as np
import polyscope as ps

np.random.seed(0)
th = 2*np.pi*np.random.rand(200,1)
P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)

C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
V,Q = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)

ps.init()
ps.register_surface_mesh("test quadtree",V,Q,edge_width=1)
ps.set_navigation_style('planar')
ps.show()