from context import gpytoolbox
import numpy as np
from scipy.sparse import csr_matrix
import igl
import polyscope as ps

np.random.seed(0)
th = 2*np.pi*np.random.rand(100,1)
P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)

C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=6,min_depth=2,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
V,Q = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)


L, stored_at = gpytoolbox.quadtree_laplacian(C,W,CH,D,A)
fun = stored_at[:,0]**2.0
# This will never be exactly two everywhere, but it should at least be two in most places. 
assert(np.isclose(np.median(np.abs(L @ fun)),2.0))


# This is not very satisfying so just to be safe let's solve a Poisson equation
th = 2*np.pi*np.random.rand(500,1)
P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)
C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=9,min_depth=5,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
V,Q = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)
L, stored_at = gpytoolbox.quadtree_laplacian(C,W,CH,D,A)
gt_fun = stored_at[:,0]**2.0
lap_fun = 2.0 + 0.0*stored_at[:,0]
bb = ((stored_at[:,0]>0.8) | (stored_at[:,0]<-0.8) | (stored_at[:,1]>0.8) | (stored_at[:,1]<-0.8)).nonzero()[0]
bc = gt_fun[bb]
Aeq = csr_matrix((0, 0), dtype=np.float64)
Beq = np.array([], dtype=np.float64)
u = igl.min_quad_with_fixed(L,-1.0*lap_fun,bb,bc,Aeq,Beq,False)[1]

# Plot solution:
# ps.init()
# quadtree = ps.register_surface_mesh("test quadtree",V,Q,edge_width=1)
# quadtree.add_scalar_quantity("numeric solve",u,defined_on='faces')
# quadtree.add_scalar_quantity("groundtruth",gt_fun,defined_on='faces')
# ps.show()

# Error will exist, since the lowest deapth leaf nodes are big, but it should be "reasonable"
assert(np.max(np.abs(u-gt_fun))<0.05)
# It should also go down if we increase the minimum depth
C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=9,min_depth=6,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
L, stored_at = gpytoolbox.quadtree_laplacian(C,W,CH,D,A)
gt_fun = stored_at[:,0]**2.0
lap_fun = 2.0 + 0.0*stored_at[:,0]
bb = ((stored_at[:,0]>0.8) | (stored_at[:,0]<-0.8) | (stored_at[:,1]>0.8) | (stored_at[:,1]<-0.8)).nonzero()[0]
bc = gt_fun[bb]
u = igl.min_quad_with_fixed(L,-1.0*lap_fun,bb,bc,Aeq,Beq,False)[1]
assert(np.max(np.abs(u-gt_fun))<0.01)
C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=9,min_depth=7,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
L, stored_at = gpytoolbox.quadtree_laplacian(C,W,CH,D,A)
gt_fun = stored_at[:,0]**2.0
lap_fun = 2.0 + 0.0*stored_at[:,0]
bb = ((stored_at[:,0]>0.8) | (stored_at[:,0]<-0.8) | (stored_at[:,1]>0.8) | (stored_at[:,1]<-0.8)).nonzero()[0]
bc = gt_fun[bb]
u = igl.min_quad_with_fixed(L,-1.0*lap_fun,bb,bc,Aeq,Beq,False)[1]
assert(np.max(np.abs(u-gt_fun))<0.005)
# *shrug*. I don't really know how to evaluate this Laplacian, especially given some error is always expected since the stencil is not exact...


# Does it *at least* converge if we force a regular grid? This will catch factors of two and minus sign errors:
C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=6,min_depth=6,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
L, stored_at = gpytoolbox.quadtree_laplacian(C,W,CH,D,A)
gt_fun = stored_at[:,0]**2.0
lap_fun = 2.0 + 0.0*stored_at[:,0]
bb = ((stored_at[:,0]>0.8) | (stored_at[:,0]<-0.8) | (stored_at[:,1]>0.8) | (stored_at[:,1]<-0.8)).nonzero()[0]
bc = gt_fun[bb]
u = igl.min_quad_with_fixed(L,-1.0*lap_fun,bb,bc,Aeq,Beq,False)[1]
assert(np.isclose(np.max(np.abs(u-gt_fun)),0.0))